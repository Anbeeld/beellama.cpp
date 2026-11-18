#include "sinq-parameters.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace sinq {
static void check(bool ok, const std::string & message) {
    if (!ok) { throw std::runtime_error("SINQ: " + message); }
}
static void fields(const nlohmann::json & j, const std::set<std::string> & allowed) {
    check(j.is_object(), "expected object");
    for (auto it=j.begin(); it!=j.end(); ++it) { check(allowed.count(it.key()) != 0, "unknown field: " + it.key()); }
}
nlohmann::json parse_json_strict(const std::string & text) {
    std::vector<std::set<std::string>> objects;
    auto callback = [&](int, nlohmann::json::parse_event_t event, nlohmann::json & value) {
        using event_t = nlohmann::json::parse_event_t;
        if (event == event_t::object_start) { objects.emplace_back(); }
        if (event == event_t::key) { check(objects.back().insert(value.get<std::string>()).second, "duplicate JSON key"); }
        if (event == event_t::object_end) { objects.pop_back(); }
        if (event == event_t::value && value.is_number_float()) { check(std::isfinite(value.get<double>()), "nonfinite JSON number"); }
        return true;
    };
    return nlohmann::json::parse(text,callback);
}
static int integer(const nlohmann::json & j, const std::string & name) {
    check(j.is_number_integer(), name + " must be an integer");
    const auto n=j.get<int64_t>();
    check(n>=0 && n<=INT32_MAX, name + " out of range");
    return int(n);
}
static double unit(const nlohmann::json & j, const std::string & name) {
    check(j.is_number(), name + " must be numeric");
    const double n=j.get<double>();
    check(std::isfinite(n) && n>=0 && n<=1, name + " must be finite in [0,1]");
    return n;
}
parameters parse_parameters(const nlohmann::json & j, bool complete) {
    fields(j,{"generator","strip_width","iterations","strength","correction","alpha","lambda"});
    if (complete) for (const auto * key : {"generator","strip_width","iterations","strength"}) {
        check(j.contains(key), std::string("missing complete parameter: ")+key);
    }
    parameters p;
    if (j.contains("generator")) { p.generator=j.at("generator").get<std::string>(); }
    check(p.generator=="sinq" || p.generator=="awq", "unsupported generator");
    if (j.contains("strip_width")) { p.strip_width=integer(j.at("strip_width"),"strip_width"); }
    if (j.contains("iterations")) { p.iterations=integer(j.at("iterations"),"iterations"); }
    if (j.contains("strength")) { p.strength=unit(j.at("strength"),"strength"); }
    check(llama_sinq_valid_strip_width(p.strip_width),"unsupported strip_width");
    check(p.iterations==4 || p.iterations==8 || p.iterations==16,"unsupported iterations");
    if (p.generator=="awq") {
        check(j.contains("correction") && j.contains("alpha") && j.contains("lambda"),"AWQ requires correction, alpha and lambda");
        p.correction=j.at("correction").get<std::string>();
        check(p.correction=="weight_activation" || p.correction=="activation_only","unsupported correction");
        p.alpha=unit(j.at("alpha"),"alpha"); p.lambda=unit(j.at("lambda"),"lambda");
    } else {
        check(!j.contains("correction") && !j.contains("alpha") && !j.contains("lambda"),"ordinary SINQ does not accept AWQ parameters");
    }
    return p;
}
nlohmann::json parameters_json(const parameters & p) {
    nlohmann::json j={{"generator",p.generator},{"strip_width",p.strip_width},{"iterations",p.iterations},{"strength",p.strength}};
    if (p.generator=="awq") { j["correction"]=p.correction; j["alpha"]=p.alpha; j["lambda"]=p.lambda; }
    parse_parameters(j);
    return j;
}
llama_sinq_options parameters::normalization() const {
    parse_parameters(parameters_json(*this));
    llama_sinq_options options;
    options.strip_width=strip_width; options.iterations=iterations; options.strength=float(strength);
    return options;
}
parameters validate_recipe(const std::string & text) {
    check(text.size()<=4096,"recipe exceeds 4096 bytes");
    const auto p=parse_parameters(parse_json_strict(text));
    check(parameters_json(p).dump()==text,"recipe must be canonical complete JSON");
    return p;
}
static std::set<std::string> names(const nlohmann::json & j) {
    check(j.is_array(),"tensor names must be an array");
    std::set<std::string> result;
    for (const auto & value:j) {
        check(value.is_string() && !value.get<std::string>().empty(),"invalid tensor name");
        check(result.insert(value.get<std::string>()).second,"duplicate tensor name");
    }
    return result;
}
std::vector<resolved_tensor> resolve_direct(const nlohmann::json & config, const std::vector<tensor_descriptor> & tensors,
                                          encoding_options * encoding) {
    fields(config,{"schema_version","defaults","targets","overrides","encoding"});
    encoding_options options;
    if (config.contains("encoding")) {
        const auto & value=config.at("encoding");
        fields(value,{"scale_refinement_lambda","imatrix_exponent"});
        if (value.contains("scale_refinement_lambda")) {
            options.scale_refinement_lambda=unit(value.at("scale_refinement_lambda"),"scale_refinement_lambda");
        }
        if (value.contains("imatrix_exponent")) {
            check(value.at("imatrix_exponent").is_number(),"imatrix_exponent must be numeric");
            options.imatrix_exponent=value.at("imatrix_exponent").get<double>();
            check(std::isfinite(options.imatrix_exponent),"imatrix_exponent must be finite");
        }
    }
    check(config.contains("schema_version") && config.at("schema_version").is_number_integer() && config.at("schema_version")==2,"unsupported config schema version");
    const auto defaults=parse_parameters(config.at("defaults"),false);
    check(defaults.generator=="sinq","direct configuration supports ordinary SINQ only");
    const auto & targets=config.at("targets"); fields(targets,{"include","exclude"});
    const auto & include=targets.at("include");
    const bool all=include.is_string() && include=="eligible";
    auto requested=all ? std::set<std::string>{} : names(include);
    check(all || !requested.empty(),"include cannot be empty");
    auto excluded=targets.contains("exclude") ? names(targets.at("exclude")) : std::set<std::string>{};
    std::vector<resolved_tensor> result;
    std::set<std::string> inventory;
    for (const auto & tensor:tensors) {
        check(inventory.insert(tensor.name).second,"duplicate tensor inventory name");
        const bool explicit_target=requested.erase(tensor.name)!=0;
        if (explicit_target) { check(tensor.eligible,"explicit target is ineligible: "+tensor.name); }
        const bool omit=excluded.erase(tensor.name)!=0;
        if ((all || explicit_target) && tensor.eligible && !omit) { result.push_back({tensor,defaults,true}); }
    }
    check(requested.empty(),"unknown explicit target"); check(excluded.empty(),"unknown exclusion");
    check(!result.empty(),"no eligible tensors selected");
    std::set<std::string> overridden;
    if (config.contains("overrides")) {
        check(config.at("overrides").is_array(),"overrides must be array");
        for (const auto & rule:config.at("overrides")) {
            fields(rule,{"selector","parameters"});
            const auto & selector=rule.at("selector"); fields(selector,{"tensor","layer","role"});
            check(selector.size()==1,"selector must have exactly one field");
            const auto & partial=rule.at("parameters");
            fields(partial,{"generator","strip_width","iterations","strength"});
            auto merged=parameters_json(defaults);
            for (auto it=partial.begin();it!=partial.end();++it) { merged[it.key()]=it.value(); }
            const auto p=parse_parameters(merged);
            check(p.generator=="sinq","direct override supports ordinary SINQ only");
            int layer=-1; std::string name,role;
            if (selector.contains("layer")) { layer=integer(selector.at("layer"),"layer"); }
            if (selector.contains("tensor")) { name=selector.at("tensor").get<std::string>(); }
            if (selector.contains("role")) { role=selector.at("role").get<std::string>(); }
            size_t matches=0;
            for (auto & item:result) {
                if ((selector.contains("layer") && item.tensor.layer==layer) ||
                    (selector.contains("tensor") && item.tensor.name==name) ||
                    (selector.contains("role") && item.tensor.role==role)) {
                    check(overridden.insert(item.tensor.name).second,"overlapping explicit overrides: "+item.tensor.name);
                    item.settings=p; ++matches;
                }
            }
            check(matches>0,"selector matched no selected eligible tensors");
        }
    }
    for (const auto & item:result) {
        const auto & t=item.tensor;
        check(t.rows>=2 && t.columns>0 && llama_sinq_supported_type(t.qtype),"unsupported selected shape/codec: "+t.name);
        check(t.columns%ggml_blck_size(t.qtype)==0 && t.columns%item.settings.strip_width==0,"selected tensor alignment: "+t.name);
        size_t workspace;
        check(item.settings.strength==0 || llama_sinq_workspace_size(t.rows,item.settings.strip_width,workspace),"selected tensor exceeds workspace: "+t.name);
    }
    std::sort(result.begin(),result.end(),[](const resolved_tensor & a,const resolved_tensor & b){return a.tensor.name<b.tensor.name;});
    if (encoding) { *encoding=options; }
    return result;
}
std::vector<resolved_tensor> resolve_direct_file(const std::string & path,const std::vector<tensor_descriptor> & tensors,
                                               encoding_options * encoding) {
    std::ifstream stream(path,std::ios::binary); check(bool(stream),"cannot open configuration: "+path);
    const std::string text((std::istreambuf_iterator<char>(stream)),std::istreambuf_iterator<char>());
    return resolve_direct(parse_json_strict(text),tensors,encoding);
}
std::vector<parameters> validate_metadata(const std::vector<std::string> & names,const std::vector<std::string> & recipes,const std::string & producer) {
    check(!names.empty() && names.size()<=100000 && names.size()==recipes.size(),"invalid metadata coverage");
    check(!producer.empty() && producer.size()<=1024,"invalid producer identity");
    size_t bytes=0; std::vector<parameters> result;
    for (size_t i=0;i<names.size();++i) {
        check(!names[i].empty() && (i==0 || names[i-1]<names[i]),"tensor names must be strictly sorted");
        bytes+=names[i].size()+recipes[i].size(); check(bytes<=16*1024*1024,"metadata exceeds 16 MiB");
        result.push_back(validate_recipe(recipes[i]));
    }
    return result;
}
}
