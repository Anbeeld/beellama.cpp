#include "sinq-candidates.h"
#include "sinq-awq.h"
#include "sinq-native-codec.h"
#include "sinq-local-score.h"
#include "sinq-numeric.h"
#include "sinq-parallel.h"
#include <algorithm>
#include <cmath>
namespace sinq {
search_result materialize_candidates(ggml_type codec,int64_t n,int64_t k,
        const std::vector<float>&x,int samples,const std::vector<float>&validation,int validation_samples,
        const float*importance,size_t limit,const weight_reader&read,const native_encoder&encode,
        const candidate_writer&emit,const scale_computer&compute,const search_options&options) {
    require(llama_sinq_supported_type(codec)&&n>=2&&k>0&&k%ggml_blck_size(codec)==0&&read&&encode&&emit&&compute,"invalid materialization codec/shape/callback");
    require(uint64_t(k)<=SIZE_MAX/1024&&uint64_t(n)<=SIZE_MAX/uint64_t(k)/4,"materialization shape overflow");
    require(options.threads>0&&options.max_candidates>0&&options.max_candidates<=1024&&!options.candidates.empty()&&options.candidates.size()<=options.max_candidates,"invalid candidate budget/threads");
    require(options.objective==local_objective::output_l1||options.objective==local_objective::output_mse,"invalid objective");
    require(samples>=0&&samples<=256&&x.size()==size_t(k)*samples,"activation cardinality mismatch");
    require(options.held_out?(samples>0&&validation_samples>0&&validation_samples<=256&&validation.size()==size_t(k)*validation_samples):(validation.empty()&&validation_samples==0),"invalid heldout activations");
    for(float v:x)require(std::isfinite(v),"nonfinite activation");
    for(float v:validation)require(std::isfinite(v),"nonfinite validation activation");
    require(importance||!ggml_quantize_requires_imatrix(codec),"codec requires imatrix");
    for(int64_t j=0;importance&&j<k;++j)require(std::isfinite(importance[j])&&importance[j]>=0,"invalid importance");
    // Validate every complete specification before encoding anything.
    for(const auto&s:options.candidates) {
        require((s.generator==generator_kind::sinq||s.generator==generator_kind::awq)&&llama_sinq_valid_strip_width(s.strip_width)&&k%s.strip_width==0&&(s.iterations==4||s.iterations==8||s.iterations==16)&&std::isfinite(s.strength)&&s.strength>=0&&s.strength<=1,"invalid candidate normalization");
        if(s.generator==generator_kind::awq)require(samples>0&&(s.correction==correction_kind::weight_activation||s.correction==correction_kind::activation_only)&&std::isfinite(s.alpha)&&s.alpha>=0&&s.alpha<=1&&std::isfinite(s.lambda)&&s.lambda>=0&&s.lambda<=1,"AWQ requires activations and valid correction parameters");
    }
    const size_t fixed=size_t(k)*(96+size_t(samples+validation_samples)*4+options.candidates.size()*2)+options.candidates.size()*256+4096;
    const size_t row_bytes=ggml_row_size(codec,k),per_row=size_t(k)*12+row_bytes+2*sizeof(double),scratch=size_t(k)*sizeof(double);
    require(limit>fixed&&limit-fixed>=per_row+scratch,"workspace cannot fit one row");
    const int threads=int(std::min<size_t>(size_t(std::min<int64_t>(n,options.threads)),(limit-fixed-per_row)/scratch));
    const size_t resident=fixed+size_t(threads)*scratch;
    const int64_t slab=std::min<int64_t>(n,(limit-resident)/per_row);
    std::vector<float>w(size_t(slab)*k),pre(w.size()),dq(w.size()),im(importance?size_t(k):0);
    std::vector<uint8_t>packed(size_t(slab)*row_bytes);
    std::vector<double>train_rows(size_t(slab),0.),validation_rows(size_t(slab),0.),mx(size_t(k),0),mw(size_t(k),0);
    std::vector<sum>sums(size_t(k),sum{});
    if(samples)parallel_for(k,threads,[&](int64_t begin,int64_t end){
        for(int64_t j=begin;j<end;++j){sum v;for(int t=0;t<samples;++t)v.add(std::abs(double(x[size_t(t)*k+j])));mx[j]=v.value/samples;}
    });
    for(int64_t first=0;first<n;first+=slab) {
        const auto rows=std::min(slab,n-first);read(first,rows,w.data());
        // Each worker owns contiguous columns; every column still adds rows
        // in source order across serial slabs, with no parallel reduction.
        parallel_for(k,threads,[&](int64_t begin,int64_t end){
            for(int64_t r=0;r<rows;++r)for(int64_t j=begin;j<end;++j){require(std::isfinite(w[r*k+j]),"nonfinite source weight");sums[j].add(std::abs(double(w[r*k+j])));}
        });
    }
    parallel_for(k,threads,[&](int64_t begin,int64_t end){for(int64_t j=begin;j<end;++j)mw[j]=sums[j].value/double(n);});
    search_result result;result.workspace_bytes=resident+size_t(slab)*per_row;
    for(size_t index=0;index<options.candidates.size();++index) {
        candidate_result r;r.spec=options.candidates[index];r.scales=compensation(compute(r.spec),mx,mw,r.spec);
        require(r.scales.size()==size_t(k),"normalization cardinality mismatch");
        r.has_score=samples>0;r.has_validation=options.held_out;r.duplicate_of=index;
        // Always materialize every requested descriptor; equal payloads never erase provenance.
        if(importance)transform_importance(importance,im.data(),k,r.scales.data());
        sum train,val;
        for(int64_t first=0;first<n;first+=slab) {
            const auto rows=std::min(slab,n-first);read(first,rows,w.data());
            precondition_weights(w.data(),pre.data(),rows,k,r.scales.data(),threads);
            const size_t bytes=encode(pre.data(),packed.data(),first,rows,importance?im.data():nullptr);
            require(bytes==size_t(rows)*row_bytes&&ggml_validate_row_data(codec,packed.data(),bytes),"invalid native packed bytes");
            if(samples) {
                parallel_for(rows,threads,[&](int64_t begin,int64_t end){for(int64_t row=begin;row<end;++row)ggml_get_type_traits(codec)->to_float(packed.data()+row*row_bytes,dq.data()+row*k,k);});
                output_scores(w.data(),dq.data(),r.scales.data(),rows,k,x.data(),samples,options.objective,train_rows.data(),threads);
                if(options.held_out)output_scores(w.data(),dq.data(),r.scales.data(),rows,k,validation.data(),validation_samples,options.objective,validation_rows.data(),threads);
                for(int64_t row=0;row<rows;++row){train.add(train_rows[size_t(row)]);if(options.held_out)val.add(validation_rows[size_t(row)]);}
            }
            emit(index,r.scales,first,rows,packed.data(),bytes);
        }
        r.train_score=train.value/double(n);r.validation_score=val.value/double(n);
        require(std::isfinite(r.train_score)&&std::isfinite(r.validation_score),"score overflow");
        r.encoded=true;++result.encode_count;result.candidates.push_back(std::move(r));
    }
    return result;
}
}
