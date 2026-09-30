// CPU two-level BVH (binned SAH) with watertight ray/triangle intersection.
#pragma once
#include <cstdint>
#include <vector>

#include "spectral/kernel/material.h"
#include "spectral/kernel/scene_view.h"

namespace spectral::cpu {

struct BvhNode {
  AABB bounds;
  uint32_t left_or_first = 0;  // interior: left child index (right = left + 1); leaf: first prim
  uint32_t count = 0;          // > 0 for leaves
};

class Bvh {
 public:
  void build(const std::vector<AABB>& prim_bounds);
  const std::vector<BvhNode>& nodes() const { return nodes_; }
  const std::vector<uint32_t>& prims() const { return prims_; }
  bool empty() const { return nodes_.empty(); }

 private:
  void subdivide(uint32_t node, const std::vector<AABB>& bounds, const std::vector<Vec3>& centroids, int depth);
  std::vector<BvhNode> nodes_;
  std::vector<uint32_t> prims_;
};

// Acceleration structure for a whole scene: one BLAS per mesh, TLAS over instances.
struct CpuAccel {
  std::vector<Bvh> blas;  // indexed by mesh
  std::vector<AABB> mesh_bounds;
  std::vector<char> blas_built;
  Bvh tlas;               // over instances
  void build(const SceneView& s);
  // Builds BLASes for meshes that have none yet and rebuilds the TLAS (after transform /
  // visibility changes or newly added meshes). Existing BLASes are kept.
  void update(const SceneView& s, uint32_t n_meshes);
};

struct CpuIntersector {
  const SceneView* scene = nullptr;
  const CpuAccel* accel = nullptr;
  bool closest(const Ray& r, float tmax, Hit* hit) const;
  bool occluded(const Ray& r, float tmax) const;

 private:
  template <bool AnyHit>
  bool traverse(const Ray& r, float tmax, Hit* hit) const;
};

}  // namespace spectral::cpu
