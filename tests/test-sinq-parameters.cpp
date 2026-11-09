#include "sinq/sinq-parameters.h"
// Keep parser assertions active in Release test builds.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cstdio>
#include <functional>
#include <limits>
#include <stdexcept>

static void rejects(const std::function<void()> & f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    if (!rejected) { throw std::runtime_error("invalid SINQ input accepted"); }
}
int main() try {
    using namespace sinq;
    rejects([] { parse_json_strict("{\"a\":1,\"a\":2}"); });
    rejects([] { parse_json_strict("{\"a\":1e999}"); });
    rejects([] { parse_parameters(parse_json_strict("{\"generator\":\"sinq\"}")); });
    rejects([] { parse_parameters(parse_json_strict("{\"generator\":\"sinq\",\"strip_width\":64,\"iterations\":16,\"strength\":1,\"unknown\":0}")); });
    const auto config = parse_json_strict(R"({"schema_version":2,"defaults":{"generator":"sinq"},"targets":{"include":"eligible"},"overrides":[{"selector":{"layer":1},"parameters":{"strip_width":32,"strength":0.5}}]})");
    std::vector<tensor_descriptor> tensors = {
        {"blk.0.attn_q.weight","attn_q",0,256,4,GGML_TYPE_Q4_K,true},
        {"blk.1.ffn_up.weight","ffn_up",1,256,4,GGML_TYPE_Q4_K,true}};
    const auto plan = resolve_direct(config,tensors);
    assert(plan.size()==2 && plan[0].settings.strip_width==64 && plan[1].settings.strip_width==32);
    assert(plan[1].settings.strength==0.5 && plan[0].locked);
    auto encoding_config=config;
    encoding_config["encoding"]={{"scale_refinement_lambda",0.25},{"imatrix_exponent",1.5}};
    encoding_options encoding;
    const auto encoding_plan=resolve_direct(encoding_config,tensors,&encoding);
    assert(encoding.scale_refinement_lambda==0.25 && encoding.imatrix_exponent==1.5);
    assert(encoding_plan.size()==plan.size());
    for (size_t i=0;i<plan.size();++i) {
        assert(parameters_json(encoding_plan[i].settings)==parameters_json(plan[i].settings));
    }
    resolve_direct(config,tensors,&encoding);
    assert(encoding.is_default());
    auto explicit_default=config;
    explicit_default["encoding"]={{"scale_refinement_lambda",0.0},{"imatrix_exponent",2.0}};
    resolve_direct(explicit_default,tensors,&encoding);
    assert(encoding.is_default());
    for (const auto & invalid : std::vector<nlohmann::json>{
             nullptr, 1, {{"lambda",0.25}}, {{"imatrix_exponent","2"}}, {{"imatrix_exponent",true}},
             {{"scale_refinement_lambda",-0.1}}, {{"scale_refinement_lambda",1.01}}, {{"scale_refinement_lambda","0.25"}},
             {{"scale_refinement_lambda",true}}}) {
        auto malformed=config; malformed["encoding"]=invalid;
        rejects([&] { resolve_direct(malformed,tensors); });
    }
    for (double invalid : {std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        for (const char * key : {"imatrix_exponent","scale_refinement_lambda"}) {
            auto malformed=config; malformed["encoding"]={{key,invalid}};
            rejects([&] { resolve_direct(malformed,tensors); });
        }
    }
    for (double exponent : {0.0,-1.0,1.73,2.0,2.12,3.0}) {
        auto valid=config; valid["encoding"]={{"imatrix_exponent",exponent}};
        resolve_direct(valid,tensors,&encoding); assert(encoding.imatrix_exponent==exponent);
    }
    auto bad=config;
    bad["overrides"].push_back({{"selector",{{"tensor",tensors[1].name}}},{"parameters",{{"strength",0.5}}}});
    rejects([&] { resolve_direct(bad,tensors); });
    bad=config; bad["targets"]["include"]={"missing"};
    rejects([&] { resolve_direct(bad,tensors); });
    bad=config; bad["overrides"][0]["parameters"]["strip_width"]=512;
    rejects([&] { resolve_direct(bad,tensors); });
    std::vector<std::string> names,recipes;
    for (const auto & item:plan) { names.push_back(item.tensor.name); recipes.push_back(parameters_json(item.settings).dump()); }
    assert(validate_metadata(names,recipes,"test").size()==2);
    rejects([&] { validate_recipe(" "+recipes[0]); });
    rejects([&] { validate_metadata({names[1],names[0]},recipes,"test"); });
    rejects([&] { validate_metadata(names,{recipes[0]},"test"); });
    rejects([&] { validate_metadata(names,recipes,""); });
    for (int width : {8,16,32,64,128,256,512}) for (int iterations : {4,8,16}) for (double strength : {0.,0.5,1.}) {
        parameters p; p.strip_width=width; p.iterations=iterations; p.strength=strength;
        assert(validate_recipe(parameters_json(p).dump()).strip_width==width);
    }
} catch (const std::exception & error) {
    std::fprintf(stderr,"%s\n",error.what());
    return 1;
}
