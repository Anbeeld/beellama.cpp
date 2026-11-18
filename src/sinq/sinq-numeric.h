#pragma once
#include <stdexcept>
namespace sinq {
inline void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
struct sum {
    double value=0, correction=0;
    void add(double x) { const double y=x-correction,next=value+y;correction=(next-value)-y;value=next; }
};
}
