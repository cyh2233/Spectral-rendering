// Sampling routines (all pdfs in solid angle unless noted).
#pragma once
#include "spectral/kernel/math.h"

namespace spectral {

SPECTRAL_FN Vec2 sample_concentric_disk(Vec2 u) {
  float ox = 2.f * u.x - 1.f, oy = 2.f * u.y - 1.f;
  if (ox == 0.f && oy == 0.f) return {0.f, 0.f};
  float r, theta;
  if (fabsf(ox) > fabsf(oy)) {
    r = ox;
    theta = (kPi / 4.f) * (oy / ox);
  } else {
    r = oy;
    theta = kPi / 2.f - (kPi / 4.f) * (ox / oy);
  }
  return {r * cosf(theta), r * sinf(theta)};
}

// Cosine-weighted hemisphere around +z (local frame).
SPECTRAL_FN Vec3 sample_cosine_hemisphere(Vec2 u) {
  Vec2 d = sample_concentric_disk(u);
  float z = safe_sqrt(1.f - d.x * d.x - d.y * d.y);
  return {d.x, d.y, z};
}
SPECTRAL_FN float cosine_hemisphere_pdf(float cos_theta) { return cos_theta > 0.f ? cos_theta * kInvPi : 0.f; }

SPECTRAL_FN Vec3 sample_uniform_sphere(Vec2 u) {
  float z = 1.f - 2.f * u.x;
  float r = safe_sqrt(1.f - z * z);
  float phi = 2.f * kPi * u.y;
  return {r * cosf(phi), r * sinf(phi), z};
}
SPECTRAL_FN float uniform_sphere_pdf() { return 1.f / (4.f * kPi); }

// Uniform cone around +z with cos of half-angle cos_max.
SPECTRAL_FN Vec3 sample_uniform_cone(Vec2 u, float cos_max) {
  float cos_t = (1.f - u.x) + u.x * cos_max;
  float sin_t = safe_sqrt(1.f - cos_t * cos_t);
  float phi = u.y * 2.f * kPi;
  return {cosf(phi) * sin_t, sinf(phi) * sin_t, cos_t};
}
SPECTRAL_FN float uniform_cone_pdf(float cos_max) { return 1.f / (2.f * kPi * (1.f - cos_max)); }

// Uniform barycentrics on a triangle (returns b1, b2; b0 = 1-b1-b2).
SPECTRAL_FN Vec2 sample_uniform_triangle(Vec2 u) {
  float b0, b1;
  if (u.x < u.y) {
    b0 = u.x / 2.f;
    b1 = u.y - b0;
  } else {
    b1 = u.y / 2.f;
    b0 = u.x - b1;
  }
  return {b1, 1.f - b0 - b1};
}

SPECTRAL_FN float power_heuristic(float fpdf, float gpdf) {
  float f = fpdf * fpdf, g = gpdf * gpdf;
  if (f == 0.f && g == 0.f) return 0.f;
  if (f > 3.0e38f) return 1.f;  // guard against inf*inf
  return f / (f + g);
}

// ---- GGX / Trowbridge-Reitz microfacet distribution (local frame, z = normal) ----
struct GGX {
  float alpha;  // isotropic
  SPECTRAL_FN float D(Vec3 h) const {
    float c2 = h.z * h.z;
    if (c2 <= 0.f) return 0.f;
    float a2 = alpha * alpha;
    float d = c2 * (a2 - 1.f) + 1.f;
    return a2 / (kPi * d * d);
  }
  SPECTRAL_FN float lambda(Vec3 w) const {
    float c2 = w.z * w.z;
    if (c2 <= 0.f) return 0.f;
    float t2 = (1.f - c2) / c2;
    return 0.5f * (sqrtf(1.f + alpha * alpha * t2) - 1.f);
  }
  SPECTRAL_FN float G1(Vec3 w) const { return 1.f / (1.f + lambda(w)); }
  // Height-correlated Smith masking-shadowing.
  SPECTRAL_FN float G(Vec3 wo, Vec3 wi) const { return 1.f / (1.f + lambda(wo) + lambda(wi)); }
  // Visible normal pdf: D_wo(h) = G1(wo) max(0, wo.h) D(h) / cos(wo).
  SPECTRAL_FN float pdf_visible(Vec3 wo, Vec3 h) const {
    if (wo.z <= 0.f) return 0.f;
    return G1(wo) * maxf(0.f, dot(wo, h)) * D(h) / wo.z;
  }
  // Heitz 2018 VNDF sampling. wo in upper hemisphere.
  SPECTRAL_FN Vec3 sample_visible(Vec3 wo, Vec2 u) const {
    Vec3 vh = normalize(Vec3(alpha * wo.x, alpha * wo.y, wo.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    Vec3 t1 = lensq > 0.f ? Vec3(-vh.y, vh.x, 0.f) / sqrtf(lensq) : Vec3(1.f, 0.f, 0.f);
    Vec3 t2 = cross(vh, t1);
    float r = sqrtf(u.x);
    float phi = 2.f * kPi * u.y;
    float p1 = r * cosf(phi), p2 = r * sinf(phi);
    float s = 0.5f * (1.f + vh.z);
    p2 = (1.f - s) * safe_sqrt(1.f - p1 * p1) + s * p2;
    Vec3 nh = t1 * p1 + t2 * p2 + vh * safe_sqrt(1.f - p1 * p1 - p2 * p2);
    return normalize(Vec3(alpha * nh.x, alpha * nh.y, maxf(1e-6f, nh.z)));
  }
};

// ---- Discrete distribution via alias table (device view) ----
struct AliasEntry {
  float prob;     // probability of keeping this bin
  uint32_t alias; // alternative bin
  float pmf;      // normalized probability of the bin
};

SPECTRAL_FN uint32_t sample_alias(const AliasEntry* table, uint32_t n, float u, float* pmf_out) {
  float x = u * float(n);
  uint32_t i = uint32_t(x);
  if (i >= n) i = n - 1;
  float up = x - float(i);
  uint32_t k = (up < table[i].prob) ? i : table[i].alias;
  if (pmf_out) *pmf_out = table[k].pmf;
  return k;
}

// ---- Piecewise-constant 1D/2D distributions (device views over flat arrays) ----
// cdf has n+1 entries with cdf[0]=0, cdf[n]=1; func_int is the integral of the function over [0,1].
SPECTRAL_FN int find_interval(const float* cdf, int n_plus_1, float u) {
  int lo = 0, hi = n_plus_1 - 1;
  while (lo + 1 < hi) {
    int mid = (lo + hi) >> 1;
    if (cdf[mid] <= u) lo = mid; else hi = mid;
  }
  return lo;
}

struct Distribution2DView {
  int nu = 0, nv = 0;             // columns (u / phi), rows (v / theta)
  const float* func = nullptr;    // nv * nu function values
  const float* cond_cdf = nullptr;// nv * (nu + 1)
  const float* row_int = nullptr; // nv integrals of each row
  const float* marg_cdf = nullptr;// nv + 1
  float marg_int = 0.f;           // integral of the whole function over [0,1]^2

  // Returns (u,v) in [0,1)^2 and pdf with respect to area on [0,1]^2.
  SPECTRAL_FN Vec2 sample(Vec2 u, float* pdf) const {
    int row = find_interval(marg_cdf, nv + 1, u.y);
    float dv = u.y - marg_cdf[row];
    float wv = marg_cdf[row + 1] - marg_cdf[row];
    float v = (float(row) + (wv > 0.f ? dv / wv : 0.f)) / float(nv);
    const float* ccdf = cond_cdf + size_t(row) * (nu + 1);
    int col = find_interval(ccdf, nu + 1, u.x);
    float du = u.x - ccdf[col];
    float wu = ccdf[col + 1] - ccdf[col];
    float uu = (float(col) + (wu > 0.f ? du / wu : 0.f)) / float(nu);
    *pdf = marg_int > 0.f ? func[size_t(row) * nu + col] / marg_int : 0.f;
    return {uu, v};
  }
  SPECTRAL_FN float pdf(Vec2 p) const {
    int col = int(p.x * nu), row = int(p.y * nv);
    col = col < 0 ? 0 : (col >= nu ? nu - 1 : col);
    row = row < 0 ? 0 : (row >= nv ? nv - 1 : row);
    return marg_int > 0.f ? func[size_t(row) * nu + col] / marg_int : 0.f;
  }
};

}  // namespace spectral
