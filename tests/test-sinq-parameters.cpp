#include "sinq/sinq-parameters.h"
#include <cassert>
#include <functional>
#include <stdexcept>

static void rejects(const std::function<void()> & f) {
    bool rejected = false;
    try { f(); } catch (const std::exception &) { rejected = true; }
    if (!rejected) { throw std::runtime_error("invalid SINQ input accepted"); }
}
int main() {
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
    for (int width : {32,64,128,256,512}) for (int iterations : {4,8,16}) for (double strength : {0.,0.5,1.}) {
        parameters p; p.strip_width=width; p.iterations=iterations; p.strength=strength;
        assert(validate_recipe(parameters_json(p).dump()).strip_width==width);
    }
}
