#include "ops.h"

using namespace Ops;

Op::Op(uint32_t height, uint32_t width): out(height, width), grad(height, width), height(height), width(width) {}
