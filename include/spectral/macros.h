// Host/device portability macros shared by all kernel headers.
// Kernel headers (include/spectral/kernel, include/spectral/spectrum/spectrum.h)
// must compile under both a host C++ compiler and nvcc: no STL containers,
// no exceptions, no virtual functions.
#pragma once

#if defined(__CUDACC__)
#define SPECTRAL_HD __host__ __device__
#define SPECTRAL_DEVICE __device__
#define SPECTRAL_INLINE __forceinline__
#else
#define SPECTRAL_HD
#define SPECTRAL_DEVICE
#define SPECTRAL_INLINE inline
#endif

#define SPECTRAL_FN SPECTRAL_HD SPECTRAL_INLINE

#ifndef SPECTRAL_MAX_BANDS
#define SPECTRAL_MAX_BANDS 128
#endif

#if defined(__CUDA_ARCH__)
#define SPECTRAL_DEVICE_CODE 1
#else
#define SPECTRAL_DEVICE_CODE 0
#endif
