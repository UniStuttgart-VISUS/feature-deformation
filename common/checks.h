#pragma once

#ifdef NDEBUG
#ifdef _MSC_VER
#include <intrin.h>
#define __stop __debugbreak();
#else
#include <cstdlib>
#define __stop std::abort();
#endif

#include <iostream>
#define __ensure(check) if (!(check)) { std::cerr << "Ensure failed: " << __FILE__ << ", " << __LINE__ << std::endl; __stop }
#else
#define __ensure(check) assert(check);
#endif
