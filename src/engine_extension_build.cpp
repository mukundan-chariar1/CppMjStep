// Compile your C implementation into the C++ extension with C linkage.
extern "C" {
  // C keywords that C++ doesn't have:
  #if defined(__cplusplus)
    #ifndef restrict
    #define restrict __restrict__
    #endif
    #ifndef _Alignof
    #define _Alignof alignof
    #endif
  #endif

  #include "engine_extension/engine_derivative_parallel.h"
  #include "engine_extension/engine_derivative_parallel.c"
}
