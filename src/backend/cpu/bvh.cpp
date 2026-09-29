#include "bvh.h"

#include <algorithm>
#include <cmath>

namespace spectral::cpu {

namespace {
constexpr int kBins = 16;
constexpr uint32_t kMaxLeaf = 4;
constexpr int kMaxDepth = 60;
}  // namespace

void Bvh::build(const std::vector<AABB>& bounds) {
  nodes_.clear();
  prims_.resize(bounds.size());
  if (bounds.empty()) return;
  std::vector<Vec3> centroids(bounds.size());
  for (size_t i = 0; i < bounds.size(); ++i) {
    prims_[i] = uint32_t(i);
    centroids[i] = bounds[i].center();
  }
  nodes_.reserve(bounds.size() * 2);
  BvhNode root;
  root.left_or_first = 0;
  root.count = uint32_t(bounds.size());
  nodes_.push_back(root);
  subdivide(0, bounds, centroids, 0);
  nodes_.shrink_to_fit();
}

void Bvh::subdivide(uint32_t ni, const std::vector<AABB>& bounds, const std::vector<Vec3>& centroids, int depth) {
  BvhNode& node = nodes_[ni];
  AABB box, cbox;
  for (uint32_t i = 0; i < node.count; ++i) {
    uint32_t p = prims_[node.left_or_first + i];
    box.expand(bounds[p]);
    cbox.expand(centroids[p]);
  }
  node.bounds = box;
  if (node.count <= kMaxLeaf || depth >= kMaxDepth) return;

  // Binned SAH over the widest centroid axis candidates (all three axes).
  float best_cost = kInf;
  int best_axis = -1, best_split = -1;
  for (int axis = 0; axis < 3; ++axis) {
    float lo = cbox.lo[axis], hi = cbox.hi[axis];
    if (!(hi > lo)) continue;
    AABB bin_box[kBins];
    uint32_t bin_cnt[kBins] = {};
    float scale = float(kBins) / (hi - lo);
    for (uint32_t i = 0; i < node.count; ++i) {
      uint32_t p = prims_[node.left_or_first + i];
      int b = std::min(kBins - 1, int((centroids[p][axis] - lo) * scale));
      bin_cnt[b]++;
      bin_box[b].expand(bounds[p]);
    }
    float left_area[kBins - 1], right_area[kBins - 1];
    uint32_t left_cnt[kBins - 1], right_cnt[kBins - 1];
    AABB lb, rb;
    uint32_t lc = 0, rc = 0;
    for (int i = 0; i < kBins - 1; ++i) {
      lc += bin_cnt[i];
      lb.expand(bin_box[i]);
      left_cnt[i] = lc;
      left_area[i] = lb.area();
      rc += bin_cnt[kBins - 1 - i];
      rb.expand(bin_box[kBins - 1 - i]);
      right_cnt[kBins - 2 - i] = rc;
      right_area[kBins - 2 - i] = rb.area();
    }
    for (int i = 0; i < kBins - 1; ++i) {
      if (left_cnt[i] == 0 || right_cnt[i] == 0) continue;
      float cost = left_cnt[i] * left_area[i] + right_cnt[i] * right_area[i];
      if (cost < best_cost) {
        best_cost = cost;
        best_axis = axis;
        best_split = i;
      }
    }
  }
  float leaf_cost = float(node.count) * box.area();
  uint32_t first = node.left_or_first, count = node.count;
  uint32_t mid;
  if (best_axis < 0 || (best_cost >= leaf_cost && count <= 16)) {
    if (count <= 16) return;  // make a leaf
    // Degenerate centroids: split in the middle of the index range.
    mid = first + count / 2;
  } else {
    float lo = cbox.lo[best_axis], hi = cbox.hi[best_axis];
    float scale = float(kBins) / (hi - lo);
    auto it = std::partition(prims_.begin() + first, prims_.begin() + first + count, [&](uint32_t p) {
      int b = std::min(kBins - 1, int((centroids[p][best_axis] - lo) * scale));
      return b <= best_split;
    });
    mid = uint32_t(it - prims_.begin());
    if (mid == first || mid == first + count) mid = first + count / 2;
  }
  uint32_t left = uint32_t(nodes_.size());
  BvhNode l, r;
  l.left_or_first = first;
  l.count = mid - first;
  r.left_or_first = mid;
  r.count = first + count - mid;
  nodes_.push_back(l);
  nodes_.push_back(r);
  nodes_[ni].left_or_first = left;
  nodes_[ni].count = 0;
  subdivide(left, bounds, centroids, depth + 1);
  subdivide(left + 1, bounds, centroids, depth + 1);
}

void CpuAccel::build(const SceneView& s) {
  // Collect meshes referenced by instances.
  uint32_t n_meshes = 0;
  for (uint32_t i = 0; i < s.n_instances; ++i) n_meshes = std::max(n_meshes, s.instances[i].mesh + 1);
  blas.assign(n_meshes, Bvh());
  std::vector<AABB> mesh_bounds(n_meshes);
  std::vector<char> built(n_meshes, 0);
  for (uint32_t i = 0; i < s.n_instances; ++i) {
    uint32_t mi = s.instances[i].mesh;
    if (built[mi]) continue;
    built[mi] = 1;
    const MeshRecord& m = s.meshes[mi];
    std::vector<AABB> tri(m.tri_count);
    for (uint32_t t = 0; t < m.tri_count; ++t) {
      uint32_t idx[3];
      triangle_indices(s, m, t, idx);
      for (int k = 0; k < 3; ++k) tri[t].expand(s.positions[idx[k]]);
      mesh_bounds[mi].expand(tri[t]);
    }
    blas[mi].build(tri);
  }
  std::vector<AABB> inst_bounds(s.n_instances);
  for (uint32_t i = 0; i < s.n_instances; ++i) {
    const AABB& b = mesh_bounds[s.instances[i].mesh];
    if (b.valid()) inst_bounds[i] = transform_aabb(s.instances[i].to_world, b);
    else inst_bounds[i].expand(Vec3(kInf * 0.5f));  // empty mesh far away
  }
  tlas.build(inst_bounds);
}

namespace {

struct RayBoxData {
  Vec3 o, inv_d;
};

inline bool hit_box(const AABB& b, const RayBoxData& r, float tmax, float* tnear) {
  float t0 = 0.f, t1 = tmax;
  for (int a = 0; a < 3; ++a) {
    float ta = (b.lo[a] - r.o[a]) * r.inv_d[a];
    float tb = (b.hi[a] - r.o[a]) * r.inv_d[a];
    if (ta > tb) std::swap(ta, tb);
    // NaN-safe: (0 * inf) produces NaN when the ray lies in a slab plane.
    t0 = ta > t0 ? ta : t0;
    t1 = tb < t1 ? tb : t1;
    if (t0 > t1) return false;
  }
  *tnear = t0;
  return true;
}

inline Vec3 safe_inv(Vec3 d) {
  auto inv = [](float x) { return std::fabs(x) > 1e-30f ? 1.f / x : std::copysign(1e30f, x); };
  return {inv(d.x), inv(d.y), inv(d.z)};
}

// Watertight ray/triangle intersection (Woop, Benthin, Wald 2013).
struct WatertightRay {
  Vec3 o;
  int kx, ky, kz;
  float sx, sy, sz;
  explicit WatertightRay(const Ray& r) {
    o = r.o;
    Vec3 ad(std::fabs(r.d.x), std::fabs(r.d.y), std::fabs(r.d.z));
    kz = ad.x > ad.y ? (ad.x > ad.z ? 0 : 2) : (ad.y > ad.z ? 1 : 2);
    kx = (kz + 1) % 3;
    ky = (kx + 1) % 3;
    if (r.d[kz] < 0.f) std::swap(kx, ky);
    sx = r.d[kx] / r.d[kz];
    sy = r.d[ky] / r.d[kz];
    sz = 1.f / r.d[kz];
  }
  bool intersect(Vec3 p0, Vec3 p1, Vec3 p2, float tmax, float* t, float* b1, float* b2) const {
    Vec3 A = p0 - o, B = p1 - o, C = p2 - o;
    float Ax = A[kx] - sx * A[kz], Ay = A[ky] - sy * A[kz];
    float Bx = B[kx] - sx * B[kz], By = B[ky] - sy * B[kz];
    float Cx = C[kx] - sx * C[kz], Cy = C[ky] - sy * C[kz];
    float U = Cx * By - Cy * Bx, V = Ax * Cy - Ay * Cx, W = Bx * Ay - By * Ax;
    if (U == 0.f || V == 0.f || W == 0.f) {
      double CxBy = double(Cx) * By, CyBx = double(Cy) * Bx;
      U = float(CxBy - CyBx);
      double AxCy = double(Ax) * Cy, AyCx = double(Ay) * Cx;
      V = float(AxCy - AyCx);
      double BxAy = double(Bx) * Ay, ByAx = double(By) * Ax;
      W = float(BxAy - ByAx);
    }
    if ((U < 0.f || V < 0.f || W < 0.f) && (U > 0.f || V > 0.f || W > 0.f)) return false;
    float det = U + V + W;
    if (det == 0.f) return false;
    float Az = sz * A[kz], Bz = sz * B[kz], Cz = sz * C[kz];
    float T = U * Az + V * Bz + W * Cz;
    if (det < 0.f ? (T >= 0.f || T < tmax * det) : (T <= 0.f || T > tmax * det)) return false;
    float inv = 1.f / det;
    *t = T * inv;
    *b1 = V * inv;
    *b2 = W * inv;
    return true;
  }
};

}  // namespace

template <bool AnyHit>
bool CpuIntersector::traverse(const Ray& ray, float tmax, Hit* hit) const {
  const SceneView& s = *scene;
  const auto& tn = accel->tlas.nodes();
  if (tn.empty()) return false;
  const auto& tp = accel->tlas.prims();
  RayBoxData wr{ray.o, safe_inv(ray.d)};
  float best = tmax;
  bool found = false;
  uint32_t stack[64];
  int sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const BvhNode& node = tn[stack[--sp]];
    float tnear;
    if (!hit_box(node.bounds, wr, best, &tnear)) continue;
    if (node.count == 0) {
      stack[sp++] = node.left_or_first + 1;
      stack[sp++] = node.left_or_first;
      continue;
    }
    for (uint32_t k = 0; k < node.count; ++k) {
      uint32_t ii = tp[node.left_or_first + k];
      const InstanceRecord& inst = s.instances[ii];
      const Bvh& blas = accel->blas[inst.mesh];
      if (blas.empty()) continue;
      const MeshRecord& mesh = s.meshes[inst.mesh];
      bool masked = inst.material >= 0 && s.materials[inst.material].alpha_mode == kAlphaMask;
      Ray lr;
      lr.o = inst.to_local.point(ray.o);
      lr.d = inst.to_local.vector(ray.d);  // unnormalized: t is preserved
      RayBoxData br{lr.o, safe_inv(lr.d)};
      WatertightRay wt(lr);
      const auto& bn = blas.nodes();
      const auto& bp = blas.prims();
      uint32_t bstack[64];
      int bsp = 0;
      bstack[bsp++] = 0;
      while (bsp > 0) {
        const BvhNode& b = bn[bstack[--bsp]];
        float bt;
        if (!hit_box(b.bounds, br, best, &bt)) continue;
        if (b.count == 0) {
          // Near-first ordering.
          const BvhNode& l = bn[b.left_or_first];
          const BvhNode& r = bn[b.left_or_first + 1];
          float tl, tr;
          bool hl = hit_box(l.bounds, br, best, &tl), hr = hit_box(r.bounds, br, best, &tr);
          if (hl && hr) {
            if (tl <= tr) {
              bstack[bsp++] = b.left_or_first + 1;
              bstack[bsp++] = b.left_or_first;
            } else {
              bstack[bsp++] = b.left_or_first;
              bstack[bsp++] = b.left_or_first + 1;
            }
          } else if (hl) {
            bstack[bsp++] = b.left_or_first;
          } else if (hr) {
            bstack[bsp++] = b.left_or_first + 1;
          }
          continue;
        }
        for (uint32_t q = 0; q < b.count; ++q) {
          uint32_t prim = bp[b.left_or_first + q];
          uint32_t idx[3];
          triangle_indices(s, mesh, prim, idx);
          float t, b1, b2;
          if (!wt.intersect(s.positions[idx[0]], s.positions[idx[1]], s.positions[idx[2]], best, &t, &b1, &b2))
            continue;
          if (masked && !alpha_test(s, ii, prim, b1, b2)) continue;
          if (AnyHit) return true;
          best = t;
          found = true;
          hit->t = t;
          hit->instance = ii;
          hit->prim = prim;
          hit->b1 = b1;
          hit->b2 = b2;
        }
      }
    }
  }
  return found;
}

bool CpuIntersector::closest(const Ray& r, float tmax, Hit* hit) const { return traverse<false>(r, tmax, hit); }
bool CpuIntersector::occluded(const Ray& r, float tmax) const { return traverse<true>(r, tmax, nullptr); }

}  // namespace spectral::cpu
