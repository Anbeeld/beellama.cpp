#include "sinq-bridge-config.h"
#include "sinq-numeric.h"
#include "sinq-parameters.h"
#include <iterator>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <set>
namespace sinq {
using json=nlohmann::json;
static void keys(const json&j,const std::set<std::string>&allowed) {
    require(j.is_object(),"expected JSON object");
    for(auto it=j.begin();it!=j.end();++it)require(allowed.count(it.key())!=0,("unknown field: "+it.key()).c_str());
}
static void integer(const json&j,const char*key) {if(j.contains(key))require(j.at(key).is_number_integer(),(std::string("expected integer: ")+key).c_str());}
json parse_bridge_json(std::istream&in) {
    return parse_json_strict(std::string(std::istreambuf_iterator<char>(in),{}));
}
candidate_spec parse_candidate_spec(const json&j) {
    const auto p=parse_parameters(j,true);candidate_spec s;
    s.generator=p.generator=="sinq"?generator_kind::sinq:generator_kind::awq;
    s.strip_width=p.strip_width;s.iterations=p.iterations;s.strength=p.strength;
    s.correction=p.correction=="activation_only"?correction_kind::activation_only:correction_kind::weight_activation;
    s.alpha=p.alpha;s.lambda=p.lambda;return s;
}
json candidate_spec_json(const candidate_spec&s) {
    parameters p;p.generator=s.generator==generator_kind::sinq?"sinq":"awq";
    p.strip_width=s.strip_width;p.iterations=s.iterations;p.strength=s.strength;
    if(s.generator==generator_kind::awq){p.correction=s.correction==correction_kind::weight_activation?"weight_activation":"activation_only";p.alpha=s.alpha;p.lambda=s.lambda;}
    return parameters_json(p);
}
static ggml_type parse_type(const std::string&name) {
    for(int t=0;t<GGML_TYPE_COUNT;++t){const char*n=ggml_type_name(ggml_type(t));if(!n)continue;std::string canonical(n);std::transform(canonical.begin(),canonical.end(),canonical.begin(),[](unsigned char c){return char(std::tolower(c));});if(name==canonical)return ggml_type(t);}
    throw std::runtime_error("unknown native qtype: "+name);
}
bridge_request parse_bridge_request(const json&j) {
    require(j.is_object(),"bridge requires object");integer(j,"schema_version");require(j.at("schema_version")==2,"unsupported bridge schema version");
    bridge_request r;r.operation=j.at("operation").get<std::string>();
    require(r.operation=="generate"||r.operation=="decode"||r.operation=="stock"||r.operation=="score","unsupported bridge operation");
    std::set<std::string>allowed={"schema_version","operation","rows","columns","codec","output","threads","workspace_bytes","max_output_bytes"};
    if(r.operation=="decode")allowed.insert({"payload","scales","first_row"});
    else {
        allowed.insert({"weights","source_type"});
        if(r.operation!="score")allowed.insert("importance");
        if(r.operation=="generate"||r.operation=="score")allowed.insert({"activations","samples","validation","validation_samples","objective","held_out"});
        if(r.operation=="generate")allowed.insert({"candidates","max_candidates"});
        if(r.operation=="score")allowed.insert({"payload","scales"});
    }
    keys(j,allowed);
    for(const char*k:{"rows","columns","first_row","samples","validation_samples","threads","workspace_bytes","max_output_bytes","max_candidates"})integer(j,k);
    for(const char*k:{"workspace_bytes","max_output_bytes","max_candidates"})if(j.contains(k))require(j.at(k).get<int64_t>()>0,"budget must be positive");
    r.rows=j.at("rows");r.columns=j.at("columns");r.codec=parse_type(j.at("codec"));r.output=j.at("output").get<std::string>();
    r.source_type=parse_type(j.value("source_type",std::string("f32")));
    r.weights=j.value("weights","");r.activations=j.value("activations","");r.validation=j.value("validation","");r.importance=j.value("importance","");r.payload=j.value("payload","");r.scales=j.value("scales","");
    r.first_row=j.value("first_row",int64_t(0));r.samples=j.value("samples",0);r.validation_samples=j.value("validation_samples",0);r.threads=j.value("threads",r.threads);
    r.workspace=j.value("workspace_bytes",r.workspace);r.max_output_bytes=j.value("max_output_bytes",r.max_output_bytes);
    require(r.rows>=2&&r.columns>0&&r.columns<=1048576&&uint64_t(r.rows)<=SIZE_MAX/uint64_t(r.columns)/4&&r.threads>0&&r.threads<=1024&&r.workspace>=4096&&r.workspace<=size_t(1024)*1024*1024&&r.max_output_bytes>0&&!r.output.empty(),"invalid bridge dimensions/budgets");
    require(llama_sinq_supported_type(r.codec)&&r.columns%ggml_blck_size(r.codec)==0,"unsupported codec/shape");
    require(r.source_type==GGML_TYPE_F32||r.source_type==GGML_TYPE_F16||r.source_type==GGML_TYPE_BF16,"source must be floating");
    r.search.threads=r.threads;const std::string objective=j.value("objective",std::string("output_l1"));require(objective=="output_l1"||objective=="output_mse","unsupported objective");
    r.search.objective=objective=="output_l1"?local_objective::output_l1:local_objective::output_mse;r.search.held_out=j.value("held_out",false);
    r.search.max_candidates=j.value("max_candidates",size_t(128));require(r.search.max_candidates<=1024,"candidate budget exceeded");
    if(r.operation=="generate") {
        require(j.at("candidates").is_array()&&!j.at("candidates").empty()&&j.at("candidates").size()<=r.search.max_candidates,"invalid candidate list/budget");
        for(const auto&c:j.at("candidates")){auto s=parse_candidate_spec(c);require(r.columns%s.strip_width==0,"candidate width does not divide columns");if(s.generator==generator_kind::awq)require(!r.activations.empty(),"AWQ requires activations");r.search.candidates.push_back(s);}
    }
    require(r.samples>=0&&r.samples<=256&&((r.samples==0)==r.activations.empty()),"invalid activation input");
    require(r.search.held_out?(!r.validation.empty()&&r.validation_samples>0&&r.validation_samples<=256&&r.samples>0):(r.validation.empty()&&r.validation_samples==0),"invalid validation input");
    if(j.contains("objective"))require(r.samples>0,"objective requires activations");
    if(r.operation=="decode")require(!r.payload.empty()&&r.first_row>=0&&uint64_t(r.first_row)<=SIZE_MAX/ggml_row_size(r.codec,r.columns),"invalid decode input");
    else require(!r.weights.empty(),"missing source weights");
    if(r.operation=="score")require(!r.payload.empty()&&r.samples>0,"missing score inputs");
    return r;
}
}
