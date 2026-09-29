// Camera ray generation. v1: pinhole. Lens effects are applied downstream by the
// optics stage (spectral_optics) from PSF / distortion data.
#pragma once
#include "spectral/kernel/scene_view.h"

namespace spectral {

// (fx, fy): continuous pixel coordinates, (0,0) = top-left corner of the image.
SPECTRAL_FN Ray generate_camera_ray(const CameraParams& c, float fx, float fy) {
  float sx = (2.f * fx / float(c.width) - 1.f) * c.tan_half_fov_x;
  float sy = (1.f - 2.f * fy / float(c.height)) * c.tan_half_fov_y;
  Vec3 d_cam = normalize(Vec3(sx, sy, -1.f));
  Ray r;
  r.o = c.cam_to_world.point(Vec3(0.f, 0.f, 0.f));
  r.d = normalize(c.cam_to_world.vector(d_cam));
  return r;
}

SPECTRAL_FN Vec3 camera_forward(const CameraParams& c) { return normalize(c.cam_to_world.vector(Vec3(0.f, 0.f, -1.f))); }

}  // namespace spectral
