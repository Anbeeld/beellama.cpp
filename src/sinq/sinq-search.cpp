#include "sinq-search.h"
#include "sinq-awq.h"
#include "sinq-local-score.h"
#include "sinq-numeric.h"
#include "sinq-parallel.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <exception>
#include <thread>
namespace sinq {
llama_sinq_calibration_result legacy_search(ggml_type codec,int64_t n,int64_t k,
        const std::vector<float> & raw,const std::vector<ggml_fp16_t> & ordinary,const std::vector<float> & x,
        int samples,const float * importance,size_t limit,const llama_sinq_weight_reader & read,const llama_sinq_native_encoder & encode,int nthread) {
    require(nthread>0,"A-SINQ invalid thread count");
    require(llama_sinq_supported_type(codec) && n>=2 && k>0 && k%ggml_blck_size(codec)==0 && samples>0 && samples<=256 && read && encode,"A-SINQ unsupported native codec, dimensions or sample count");
    require(uint64_t(k)<=std::numeric_limits<size_t>::max()/256/4 && uint64_t(n)<=std::numeric_limits<size_t>::max()/uint64_t(k)/4 && raw.size()==size_t(k) && ordinary.size()==size_t(k) && x.size()==size_t(k)*samples,"A-SINQ dimension/activation cardinality mismatch");
    for (float v:x)require(std::isfinite(v),"A-SINQ nonfinite activation");
    for (int64_t j=0;j<k;++j) {
        require(raw[j]>0 && std::isfinite(raw[j]) && ggml_fp16_to_fp32(ordinary[j])>0 && std::isfinite(ggml_fp16_to_fp32(ordinary[j])) && ordinary[j]==ggml_fp32_to_fp16(raw[j]),"A-SINQ baseline scale/rounding mismatch");
        if (importance)require(importance[j]>=0 && std::isfinite(importance[j]),"A-SINQ invalid importance");
    }
    require(importance || !ggml_quantize_requires_imatrix(codec),"A-SINQ native codec requires the final quantizer's imatrix");
    const size_t row_bytes=ggml_row_size(codec,k),fixed=size_t(k)*(64+size_t(samples)*4)+4096;
    const size_t per_row=size_t(k)*12+row_bytes+sizeof(double);
    require(limit>fixed && limit-fixed>=per_row,"A-SINQ workspace limit cannot fit one native row");
    const int scoring_threads=int(std::min<size_t>(size_t(std::min<int64_t>(n,nthread)),
        1+(limit-fixed-per_row)/(size_t(k)*sizeof(double))));
    // Each scorer owns its reconstruction errors. Include them in the workspace
    // budget before choosing a slab, rather than multiplying the bounded slab.
    require(size_t(scoring_threads-1)<=(std::numeric_limits<size_t>::max()-fixed)/sizeof(double)/size_t(k),"A-SINQ thread workspace overflow");
    const size_t threaded_fixed=fixed+size_t(scoring_threads-1)*size_t(k)*sizeof(double);
    require(limit>threaded_fixed && limit-threaded_fixed>=per_row,"A-SINQ workspace limit cannot fit one native row");
    const int64_t slab=std::min<int64_t>(n,(limit-threaded_fixed)/per_row);
    const size_t cells=size_t(slab)*k;
    std::vector<float> w(cells),pre(cells),dq(cells),transformed_importance(importance ? size_t(k) : 0);
    std::vector<uint8_t> packed(size_t(slab)*row_bytes);
    std::vector<double> mx(size_t(k),0.),mw(size_t(k),0.);
    std::vector<std::vector<double>> errors(size_t(scoring_threads),std::vector<double>(size_t(k),0.));
    std::vector<double> row_losses(size_t(slab),0.);
    for (int64_t j=0;j<k;++j) { sum v;for(int s=0;s<samples;++s)v.add(std::abs(double(x[size_t(s)*k+j])));mx[j]=v.value/samples; }
    // Column means use a stable row order independent of slab boundaries.
    std::vector<sum> weight_sums(size_t(k),sum{});
    for (int64_t first=0;first<n;first+=slab) {
        const int64_t rows=std::min(slab,n-first);read(first,rows,w.data());
        for (int64_t r=0;r<rows;++r)for(int64_t j=0;j<k;++j) {
            const float v=w[r*k+j];require(std::isfinite(v),"A-SINQ nonfinite original weight");weight_sums[j].add(std::abs(double(v)));
        }
    }
    for (int64_t j=0;j<k;++j)mw[j]=weight_sums[j].value/double(n);
    llama_sinq_calibration_result result;result.scales=ordinary;result.workspace_bytes=threaded_fixed+size_t(slab)*per_row;
    for (int candidate=-1;candidate<20;++candidate) {
        std::vector<ggml_fp16_t> u=ordinary;
        if (candidate>=0) {
            const auto a=legacy_awq_factors(mx,mw,candidate);
            for(int64_t j=0;j<k;++j) {
                // dual_shift.py matrix*=a; mu1/=a. Round exactly once.
                u[j]=ggml_fp32_to_fp16(float(double(raw[j])/a[j]));
                const float v=ggml_fp16_to_fp32(u[j]);require(v>0 && std::isfinite(v),"A-SINQ candidate underflowed/overflowed its F16 companion");
            }
        }
        if (importance)for(int64_t j=0;j<k;++j) {
            const float v=ggml_fp16_to_fp32(u[j]);transformed_importance[j]=importance[j]*v*v;
            require(std::isfinite(transformed_importance[j]) && transformed_importance[j]>=0,"A-SINQ transformed imatrix overflow");
        }
        sum loss;
        for(int64_t first=0;first<n;first+=slab) {
            const int64_t rows=std::min(slab,n-first);read(first,rows,w.data());
            for(int64_t i=0;i<rows*k;++i)pre[i]=w[i]/ggml_fp16_to_fp32(u[i%k]);
            const size_t bytes=encode(pre.data(),packed.data(),first,rows,importance ? transformed_importance.data() : nullptr);
            require(bytes==size_t(rows)*row_bytes && ggml_validate_row_data(codec,packed.data(),bytes),"A-SINQ native encoder output mismatch");
            for(int64_t r=0;r<rows;++r)ggml_get_type_traits(codec)->to_float(packed.data()+r*row_bytes,dq.data()+r*k,k);
            // Independent rows retain the exact scalar Kahan operations. Reduce
            // the completed row losses in original order for bit-exact selection.
            const int use=int(std::min<int64_t>(scoring_threads,rows));
            std::vector<std::exception_ptr> failures(size_t(use),nullptr);
            const auto score_rows=[&](int worker) {
                try {
                    const int64_t begin=rows*worker/use,end=rows*(worker+1)/use;
                    for(int64_t r=begin;r<end;++r) {
                        row_losses[size_t(r)]=row_loss(w.data()+r*k,dq.data()+r*k,u.data(),k,x.data(),samples,errors[size_t(worker)]);
                    }
                } catch (...) { failures[size_t(worker)]=std::current_exception(); }
            };
            std::vector<std::thread> scorers;
            try {
                for(int worker=1;worker<use;++worker)scorers.emplace_back(score_rows,worker);
                score_rows(0);
            } catch (...) {
                for(auto & worker:scorers)worker.join();
                throw;
            }
            for(auto & worker:scorers)worker.join();
            for(const auto & failure:failures)if(failure)std::rethrow_exception(failure);
            for(int64_t r=0;r<rows;++r)loss.add(row_losses[size_t(r)]);
        }
        const double score=loss.value/double(n);require(std::isfinite(score),"A-SINQ nonfinite layer-output L1");result.candidate_losses.push_back(score);
        if(candidate<0) { result.baseline_loss=result.selected_loss=score; }
        else {
            const double tolerance=64*std::numeric_limits<double>::epsilon()*std::max({1.,score,result.selected_loss});
            if(score<result.selected_loss-tolerance) { result.selected_loss=score;result.alpha_index=candidate;result.scales=u; }
        }
    }
    return result;
}
}

namespace sinq {
search_result research_search(ggml_type codec,int64_t n,int64_t k,const std::vector<float>&raw,
        const std::vector<float>&x,int samples,const std::vector<float>&validation,int validation_samples,
        const float* importance,size_t limit,const llama_sinq_weight_reader&read,
        const llama_sinq_native_encoder&encode,const candidate_writer&emit,const search_options&options) {
    require(llama_sinq_supported_type(codec) && n>=2 && k>0 && k%ggml_blck_size(codec)==0 &&
        samples>0 && samples<=256 && read && encode && emit,"research invalid codec/shape/readers");
    require(uint64_t(k)<=SIZE_MAX/1024 && uint64_t(n)<=SIZE_MAX/uint64_t(k)/4 &&
        raw.size()==size_t(k) && x.size()==size_t(k)*samples,"research activation cardinality/shape overflow");
    require(options.threads>0,"research invalid thread count");
    require(options.objective==local_objective::output_l1 || options.objective==local_objective::output_mse,"research invalid objective");
    require(std::isfinite(options.tolerance) && options.tolerance>=0 && options.max_candidates>0 && options.max_candidates<=1024 &&
        options.max_shortlist<=options.max_candidates && std::isfinite(options.refine_step) && options.refine_step>0 && options.refine_step<=1,"research invalid search budget/policy");
    require(!options.held_out || (validation_samples>0 && validation_samples<=256 && validation.size()==size_t(k)*validation_samples),"research missing heldout activations");
    require(options.held_out || (validation.empty() && validation_samples==0),"research unused validation input");
    for(float a:x)require(std::isfinite(a),"research nonfinite training activation");
    for(float a:validation)require(std::isfinite(a),"research nonfinite validation activation");
    for(int64_t j=0;j<k;++j) {
        require(raw[j]>0 && std::isfinite(raw[j]),"research invalid raw factor");
        if(importance)require(importance[j]>=0 && std::isfinite(importance[j]),"research invalid importance");
    }
    require(importance || !ggml_quantize_requires_imatrix(codec),"research codec requires imatrix");
    // Include every retained F16 hypothesis and its metadata, not only slabs.
    const size_t base_fixed=size_t(k)*(96+size_t(samples+validation_samples)*4+options.max_candidates*2)+options.max_candidates*256+4096;
    const size_t row_bytes=ggml_row_size(codec,k),per_row=size_t(k)*12+row_bytes+2*sizeof(double);
    const size_t scratch=size_t(k)*sizeof(double);
    require(limit>base_fixed && limit-base_fixed>=per_row+scratch,"research workspace cannot fit one row");
    const int scoring_threads=int(std::min<size_t>(size_t(std::min<int64_t>(n,options.threads)),
        (limit-base_fixed-per_row)/scratch));
    const size_t fixed=base_fixed+size_t(scoring_threads)*scratch;
    require(limit>fixed && limit-fixed>=per_row,"research workspace cannot fit one row");
    const int64_t slab=std::min<int64_t>(n,(limit-fixed)/per_row);
    std::vector<float>w(size_t(slab)*k),pre(w.size()),dq(w.size()),im(importance?size_t(k):0);
    std::vector<uint8_t>packed(size_t(slab)*row_bytes);
    std::vector<double>train_rows(size_t(slab),0.),validation_rows(size_t(slab),0.);
    std::vector<double>mx(size_t(k),0),mw(size_t(k),0);
    std::vector<sum>sums(size_t(k),sum{});
    for(int64_t j=0;j<k;++j){sum v;for(int t=0;t<samples;++t)v.add(std::abs(double(x[size_t(t)*k+j])));mx[j]=v.value/samples;}
    for(int64_t first=0;first<n;first+=slab) {
        const auto rows=std::min(slab,n-first);read(first,rows,w.data());
        for(int64_t r=0;r<rows;++r)for(int64_t j=0;j<k;++j){require(std::isfinite(w[r*k+j]),"research nonfinite source weight");sums[j].add(std::abs(double(w[r*k+j])));}
    }
    for(int64_t j=0;j<k;++j)mw[j]=sums[j].value/double(n);
    std::vector<candidate_spec>specs={{scale_family::ordinary,0,0}};
    auto supplied=options.candidates;
    const auto less=[](const candidate_spec&a,const candidate_spec&b){
        if(a.family!=b.family)return a.family<b.family;
        if(a.alpha!=b.alpha)return a.alpha<b.alpha;
        return a.lambda<b.lambda;
    };
    std::sort(supplied.begin(),supplied.end(),less);
    for(auto s:supplied) {
        // Validate even canonical identities and candidates later deduplicated.
        compensation(raw,mx,mw,s);
        if(s.family==scale_family::ordinary)continue;
        specs.push_back(s);
    }
    require(specs.size()<=options.max_candidates && options.max_shortlist<=(options.max_candidates-specs.size())/2,"research candidate budget exhausted before encode");
    search_result result;result.workspace_bytes=fixed+size_t(slab)*per_row;
    const size_t coarse=specs.size();
    const auto selection_score=[&](const candidate_result&r){return options.held_out?r.validation_score:r.train_score;};
    for(size_t index=0;index<specs.size();++index) {
        candidate_result r;r.spec=specs[index];r.scales=compensation(raw,mx,mw,r.spec);r.has_validation=options.held_out;r.duplicate_of=index;
        for(size_t j=0;j<result.candidates.size();++j)if(result.candidates[j].scales==r.scales){r.duplicate_of=result.candidates[j].duplicate_of;break;}
        if(r.duplicate_of!=index) {
            const auto& prior=result.candidates[r.duplicate_of];r.train_score=prior.train_score;r.validation_score=prior.validation_score;
        } else {
            if(importance)for(int64_t j=0;j<k;++j){const float u=ggml_fp16_to_fp32(r.scales[j]);im[j]=importance[j]*u*u;require(std::isfinite(im[j]),"research transformed importance overflow");}
            sum train,val;
            for(int64_t first=0;first<n;first+=slab) {
                const auto rows=std::min(slab,n-first);read(first,rows,w.data());
                parallel_for(rows, scoring_threads, [&](int64_t begin, int64_t end) {
                    for(int64_t i=begin*k;i<end*k;++i)pre[i]=w[i]/ggml_fp16_to_fp32(r.scales[i%k]);
                });
                const size_t bytes=encode(pre.data(),packed.data(),first,rows,importance?im.data():nullptr);
                require(bytes==size_t(rows)*row_bytes && ggml_validate_row_data(codec,packed.data(),bytes),"research invalid native packed bytes");
                parallel_for(rows, scoring_threads, [&](int64_t begin, int64_t end) {
                    for(int64_t row=begin;row<end;++row)
                        ggml_get_type_traits(codec)->to_float(packed.data()+row*row_bytes,dq.data()+row*k,k);
                });
                output_scores(w.data(),dq.data(),r.scales.data(),rows,k,x.data(),samples,options.objective,train_rows.data(),scoring_threads);
                if(options.held_out)output_scores(w.data(),dq.data(),r.scales.data(),rows,k,validation.data(),validation_samples,options.objective,validation_rows.data(),scoring_threads);
                // Preserve scalar row-order Kahan reductions across threads/slabs.
                for(int64_t row=0;row<rows;++row) {
                    train.add(train_rows[size_t(row)]);
                    if(options.held_out)val.add(validation_rows[size_t(row)]);
                }
                emit(index,r.scales,first,rows,packed.data(),bytes);
            }
            r.train_score=train.value/double(n);r.validation_score=val.value/double(n);r.encoded=true;++result.encode_count;
        }
        result.candidates.push_back(r);
        const double score=selection_score(r),best=selection_score(result.candidates[result.selected]);
        const double tolerance=std::max(options.tolerance,64*std::numeric_limits<double>::epsilon()*std::max({1.,score,best}));
        require(std::isfinite(score),"research score overflow");
        if(score<best-tolerance)result.selected=index;
        if(index+1==coarse && options.max_shortlist) {
            std::vector<size_t>rank;for(size_t j=1;j<coarse;++j)if(result.candidates[j].encoded)rank.push_back(j);
            std::stable_sort(rank.begin(),rank.end(),[&](size_t a,size_t b){return selection_score(result.candidates[a])<selection_score(result.candidates[b]);});
            rank.resize(std::min(rank.size(),options.max_shortlist));
            for(auto j:rank)for(int direction:{-1,1}) {
                auto next=specs[j];next.alpha+=direction*options.refine_step;
                if(next.alpha<0 || next.alpha>1){result.skipped.push_back({next,"alpha_outside_domain"});continue;}
                bool duplicate=false;for(const auto&s:specs)if(!less(s,next)&&!less(next,s))duplicate=true;
                if(!duplicate)specs.push_back(next);
                else result.skipped.push_back({next,"duplicate_specification"});
            }
        }
    }
    return result;
}
}
