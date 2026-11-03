#include "sinq/sinq-research-config.h"
#include <cstdio>
#include <stdexcept>
using nlohmann::json;
static void check(bool v,const char*s){if(!v)throw std::runtime_error(s);}
int main()try {
    json j={{"schema_version",1},{"operation","generate"},{"weights","w"},{"activations","x"},{"samples",4},{"rows",3},{"columns",256},{"codec","q3_k"},{"output","out"}};
    check(sinq::parse_bridge_request(j).codec==GGML_TYPE_Q3_K,"case-normalized native name");
    const auto reject=[&](json value){bool failed=false;try{sinq::parse_bridge_request(value);}catch(...){failed=true;}check(failed,"invalid bridge config accepted");};
    auto bad=j;bad["codec"]="nonsense";reject(bad);
    bad=j;bad["unknown"]=1;reject(bad);
    bad=j;bad["rows"]=-1;reject(bad);
    bad=j;bad["columns"]=257;reject(bad);
    bad=j;bad["threads"]=0;reject(bad);
    bad=j;bad["held_out"]=true;reject(bad);
    bad=j;bad["candidates"]={{{"family","awq_activation_only"},{"alpha",2},{"lambda",.5}}};reject(bad);
    bad=j;bad["candidates"]={{{"family","mystery"},{"alpha",.5},{"lambda",.5}}};reject(bad);
    puts("test-sinq-research-config: PASS");return 0;
}catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());return 1;}
