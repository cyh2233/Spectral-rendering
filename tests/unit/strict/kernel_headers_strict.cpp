// Compiled with -fno-exceptions -fno-rtti -Werror: kernel headers must not depend on
// host-only facilities (STL containers, exceptions, RTTI, virtual dispatch).
#define vector DO_NOT_USE_STD_VECTOR_IN_KERNELS
#define string DO_NOT_USE_STD_STRING_IN_KERNELS
#include "spectral/kernel/integrator.h"
#include "spectral/kernel/film.h"
#undef vector
#undef string

namespace {
struct NullIsect {
  bool closest(const spectral::Ray&, float, spectral::Hit*) const { return false; }
  bool occluded(const spectral::Ray&, float) const { return false; }
};
}  // namespace

void kernel_strict_instantiate(const spectral::SceneView& s, spectral::Spectrum& L) {
  NullIsect isect;
  spectral::PathAov aov;
  spectral::render_sample(s, isect, 0, 0, 0u, L, &aov);
}
