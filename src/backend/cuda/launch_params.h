// Parameters shared between the host (cuda_backend.cpp) and the OptiX device programs.
#pragma once
#include <optix_types.h>

#include "spectral/kernel/film.h"
#include "spectral/kernel/scene_view.h"

namespace spectral::cuda {

struct LaunchParams {
  SceneView scene;  // all pointers are device pointers
  FilmView film;    // device buffers, accumulated across passes
  OptixTraversableHandle handle;
  int32_t first_sample;
  int32_t sample_count;
};

// Payload layout for radiance rays: t (float bits), instance, primitive, b1, b2 (float bits).
constexpr unsigned kNumPayloadValues = 5;
constexpr unsigned kRayTypeRadiance = 0;
constexpr unsigned kRayTypeShadow = 1;  // miss program index

}  // namespace spectral::cuda
