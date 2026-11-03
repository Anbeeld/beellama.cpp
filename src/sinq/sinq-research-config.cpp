#include "sinq-research-config.h"
#include "sinq-numeric.h"
#include <cmath>
#include <algorithm>
#include <cctype>
#include <set>
namespace sinq {
static ggml_type parse_type(const std::string & name) {
    for(int t=0;t<GGML_TYPE_COUNT;++t) {
        const char * native=ggml_type_name(ggml_type(t));
        if(!native)continue;
        std::string canonical(native);
        std::transform(canonical.begin(),canonical.end(),canonical.begin(),[](unsigned char c){return char(std::tolower(c));});
        if(name==canonical)return ggml_type(t);
    }
    throw std::runtime_error("research unknown native qtype: "+name);
}
bridge_request parse_bridge_request(const nlohmann::json & j) {
    require(j.is_object(),"research bridge requires object");
    const std::set<std::string> keys={"schema_version","operation","weights","activations","validation","importance","payload","scales","output","codec","source_type","rows","columns","first_row","samples","validation_samples","threads","workspace_bytes","max_output_bytes","strip_width","iterations","strength","candidates","objective","held_out","tolerance","max_candidates","max_shortlist","refine_step"};
    for(auto it=j.begin();it!=j.end();++it)require(keys.count(it.key())!=0,"research bridge unknown field");
    require(j.at("schema_version")==1,"research bridge schema version");
    bridge_request r;r.operation=j.at("operation").get<std::string>();
    require(r.operation=="generate" || r.operation=="decode" || r.operation=="stock" || r.operation=="score","research bridge unsupported operation");
    r.rows=j.at("rows");r.columns=j.at("columns");r.codec=parse_type(j.at("codec"));
    r.source_type=parse_type(j.value("source_type",std::string("f32")));
    r.output=j.at("output").get<std::string>();
    r.weights=j.value("weights","");r.activations=j.value("activations","");r.validation=j.value("validation","");
    r.importance=j.value("importance","");r.payload=j.value("payload","");r.scales=j.value("scales","");
    r.first_row=j.value("first_row",int64_t(0));r.samples=j.value("samples",0);r.validation_samples=j.value("validation_samples",0);
    r.threads=j.value("threads",24);r.workspace=j.value("workspace_bytes",size_t(64*1024*1024));r.max_output_bytes=j.value("max_output_bytes",size_t(64*1024*1024));
    require(r.rows>=2 && r.columns>0 && r.columns<=1048576 && uint64_t(r.rows)<=SIZE_MAX/uint64_t(r.columns)/4 &&
        r.threads>0 && r.threads<=1024 && r.workspace>=4096 && r.workspace<=size_t(1024)*1024*1024 && r.max_output_bytes>0,"research bridge invalid dimensions/budgets");
    require(llama_sinq_supported_type(r.codec) && r.columns%ggml_blck_size(r.codec)==0,"research bridge unsupported codec/shape");
    require(r.source_type==GGML_TYPE_F32 || r.source_type==GGML_TYPE_F16 || r.source_type==GGML_TYPE_BF16,"research source must be floating");
    r.normalization.strip_width=j.value("strip_width",64);r.normalization.iterations=j.value("iterations",16);r.normalization.strength=j.value("strength",1.f);
    require(llama_sinq_valid_strip_width(r.normalization.strip_width) && r.columns%r.normalization.strip_width==0 &&
        (r.normalization.iterations==4 || r.normalization.iterations==8 || r.normalization.iterations==16) && std::isfinite(r.normalization.strength) &&
        r.normalization.strength>=0 && r.normalization.strength<=1,"research invalid Sinkhorn configuration");
    const auto objective=j.value("objective",std::string("output_l1"));
    require(objective=="output_l1" || objective=="output_mse","research unsupported objective");
    r.search.objective=objective=="output_l1"?local_objective::output_l1:local_objective::output_mse;
    r.search.held_out=j.value("held_out",false);r.search.tolerance=j.value("tolerance",0.);
    r.search.max_candidates=j.value("max_candidates",size_t(128));r.search.max_shortlist=j.value("max_shortlist",size_t(0));r.search.refine_step=j.value("refine_step",.05);
    if(j.contains("candidates")) for(const auto & c:j.at("candidates")) {
        require(c.is_object() && c.size()==3 && c.contains("family") && c.contains("alpha") && c.contains("lambda"),"research invalid candidate spec");
        const std::string f=c.at("family");candidate_spec spec;
        if(f=="ordinary")spec.family=scale_family::ordinary;
        else if(f=="awq_weight_activation")spec.family=scale_family::weight_activation;
        else if(f=="awq_activation_only")spec.family=scale_family::activation_only;
        else throw std::runtime_error("research unsupported generator");
        spec.alpha=c.at("alpha");spec.lambda=c.at("lambda");
        require(std::isfinite(spec.alpha) && spec.alpha>=0 && spec.alpha<=1 && std::isfinite(spec.lambda) && spec.lambda>=0 && spec.lambda<=1,"research invalid alpha/lambda");
        r.search.candidates.push_back(spec);
    }
    require(r.search.max_candidates>0 && r.search.max_candidates<=1024 && r.search.candidates.size()<r.search.max_candidates &&
        r.search.max_shortlist<=(r.search.max_candidates-r.search.candidates.size()-1)/2,"research candidate budget exceeded");
    if(r.operation=="generate" || r.operation=="score") require(!r.weights.empty() && !r.activations.empty() && r.samples>0 && r.samples<=256 &&
        (!r.search.held_out || (!r.validation.empty() && r.validation_samples>0 && r.validation_samples<=256)),"research missing generate input");
    if(r.operation=="score") require(!r.payload.empty() && r.search.candidates.empty(),"research missing score payload");
    if(r.operation=="stock") require(!r.weights.empty() && r.activations.empty() && r.search.candidates.empty(),"research invalid stock operation");
    if(r.operation=="decode") require(!r.payload.empty() && r.weights.empty() && r.activations.empty() && r.search.candidates.empty() && r.first_row>=0,"research invalid decode input");
    return r;
}
}
