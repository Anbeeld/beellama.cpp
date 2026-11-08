#include "sinq/sinq-bridge-config.h"
#include <cstdio>
#include <sstream>
#include <stdexcept>
using json=nlohmann::json;
static void check(bool v,const char*s){if(!v)throw std::runtime_error(s);}
int main()try{
    json spec={{"generator","sinq"},{"strip_width",64},{"iterations",16},{"strength",1.}};
    json j={{"schema_version",2},{"operation","generate"},{"rows",3},{"columns",512},{"codec","q5_0"},{"weights","w"},{"output","out"},{"candidates",json::array({spec})}};
    auto r=sinq::parse_bridge_request(j);check(r.samples==0&&r.search.candidates.size()==1,"ordinary capture independent");
    for(int width:{8,16,32,64,128,256,512})for(int it:{4,8,16})for(double strength:{0.,.5,1.}){auto s=spec;s["strip_width"]=width;s["iterations"]=it;s["strength"]=strength;j["candidates"]=json::array({s});auto a=sinq::parse_bridge_request(j);check(sinq::candidate_spec_json(a.search.candidates[0])==s,"complete spec roundtrip");}
    j["candidates"]=json::array({spec});
    auto fails=[&](json bad){bool failed=false;try{sinq::parse_bridge_request(bad);}catch(...){failed=true;}check(failed,"invalid request accepted");};
    for(auto key:{"strip_width","iterations","strength","max_shortlist","refine_step","tolerance","family"}){auto bad=j;bad[key]=1;fails(bad);}
    for(auto pair:{std::make_pair("strip_width",48),std::make_pair("iterations",7),std::make_pair("strength",2)}){auto bad=j;bad["candidates"][0][pair.first]=pair.second;fails(bad);}
    for(auto key:{"generator","strip_width","iterations","strength"}){auto bad=j;bad["candidates"][0].erase(key);fails(bad);}
    {auto bad=j;bad["candidates"][0]["strip_width"]=64.;fails(bad);}
    {auto bad=j;bad["schema_version"]=1;fails(bad);}
    {auto bad=j;bad["schema_version"]=2.;fails(bad);}
    {auto bad=j;bad["max_candidates"]=-1;fails(bad);}
    {auto bad=j;bad["candidates"][0]["generator"]="awq";bad["candidates"][0]["correction"]="activation_only";bad["candidates"][0]["alpha"]=.5;bad["candidates"][0]["lambda"]=1.;fails(bad);bad["activations"]="x";bad["samples"]=2;check(sinq::parse_bridge_request(bad).samples==2,"AWQ capture");}
    for(const char*text:{"{\"a\":1,\"a\":2}","{\"a\":{\"x\":1,\"x\":2}}","{\"a\":1e999}"}){bool failed=false;try{std::istringstream in(text);sinq::parse_bridge_json(in);}catch(...){failed=true;}check(failed,"strict JSON");}
    puts("test-sinq-bridge-config: PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
