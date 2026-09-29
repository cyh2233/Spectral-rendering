#include "spectral/scene/scene_json.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>

#include "spectral/io/csv_spectrum.h"
#include "spectral/io/image_io.h"
#include "spectral/kernel/uplift.h"
#include "spectral/scene/gltf_loader.h"
#include "spectral/scene/primitives.h"
#include "spectral/sky/sky_model.h"
#include "spectral/spectrum/glass.h"
#include "spectral/spectrum/uplift_lut.h"
#include "spectral/util/paths.h"

namespace fs = std::filesystem;
using nlohmann::json;

namespace spectral {

namespace {

[[noreturn]] void fail(const std::string& where, const std::string& msg) {
  throw std::runtime_error("scene JSON: " + where + ": " + msg);
}

Vec3 vec3(const json& j, const std::string& where) {
  if (!j.is_array() || j.size() != 3) fail(where, "expected [x, y, z]");
  return Vec3(j[0].get<float>(), j[1].get<float>(), j[2].get<float>());
}

Affine matrix4(const json& j, const std::string& where) {
  if (!j.is_array() || j.size() < 3) fail(where, "expected 4x4 (or 3x4) row-major matrix");
  Affine a;
  for (int r = 0; r < 3; ++r) {
    if (!j[r].is_array() || j[r].size() != 4) fail(where, "matrix rows must have 4 entries");
    for (int c = 0; c < 4; ++c) a.m[r][c] = j[r][c].get<float>();
  }
  return a;
}

Affine quat_rotation(const json& q, const std::string& where) {
  if (!q.is_array() || q.size() != 4) fail(where, "rotation quaternion must be [x, y, z, w]");
  double x = q[0], y = q[1], z = q[2], w = q[3];
  double n = std::sqrt(x * x + y * y + z * z + w * w);
  x /= n;
  y /= n;
  z /= n;
  w /= n;
  Affine a = Affine::identity();
  double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                    {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                    {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) a.m[r][c] = float(R[r][c]);
  return a;
}

Affine euler_deg(const json& e, const std::string& where) {
  Vec3 d = vec3(e, where) * (kPi / 180.f);
  auto rx = Affine::identity(), ry = Affine::identity(), rz = Affine::identity();
  rx.m[1][1] = std::cos(d.x); rx.m[1][2] = -std::sin(d.x); rx.m[2][1] = std::sin(d.x); rx.m[2][2] = std::cos(d.x);
  ry.m[0][0] = std::cos(d.y); ry.m[0][2] = std::sin(d.y); ry.m[2][0] = -std::sin(d.y); ry.m[2][2] = std::cos(d.y);
  rz.m[0][0] = std::cos(d.z); rz.m[0][1] = -std::sin(d.z); rz.m[1][0] = std::sin(d.z); rz.m[1][1] = std::cos(d.z);
  return compose(rz, compose(ry, rx));  // extrinsic X, then Y, then Z
}

Affine parse_transform(const json& o, const std::string& where) {
  if (o.contains("transform")) return matrix4(o["transform"], where + ".transform");
  Affine t = Affine::identity(), r = Affine::identity(), s = Affine::identity();
  if (o.contains("translate")) {
    Vec3 v = vec3(o["translate"], where + ".translate");
    t.m[0][3] = v.x;
    t.m[1][3] = v.y;
    t.m[2][3] = v.z;
  }
  if (o.contains("rotation")) r = quat_rotation(o["rotation"], where + ".rotation");
  else if (o.contains("rotate_deg")) r = euler_deg(o["rotate_deg"], where + ".rotate_deg");
  if (o.contains("scale")) {
    Vec3 v = o["scale"].is_number() ? Vec3(o["scale"].get<float>()) : vec3(o["scale"], where + ".scale");
    s.m[0][0] = v.x;
    s.m[1][1] = v.y;
    s.m[2][2] = v.z;
  }
  return compose(t, compose(r, s));
}

bool regex_match_name(const std::string& pattern, const std::string& name) {
  return std::regex_search(name, std::regex(pattern));
}

class Loader {
 public:
  Loader(const json& root, std::string base) : root_(root), base_(std::move(base)) {}

  std::shared_ptr<Scene> load() {
    scene_ = std::make_shared<Scene>();
    Scene& sc = *scene_;
    const json render = root_.value("render", json::object());
    // Band grid first: every spectrum depends on it.
    if (render.contains("bands")) {
      const json& b = render["bands"];
      sc.set_band_grid(b.value("min", 380.f), b.value("max", 1000.f), b.value("step", 5.f));
    }
    parse_render(render);
    parse_output(root_.value("output", json::object()));
    parse_camera(root_.value("camera", json::object()));
    parse_materials(root_.value("materials", json::object()));
    for (const auto& [k, v] : root_.value("assets", json::object()).items()) asset_defs_[k] = v;
    const json& insts = root_.value("instances", json::array());
    for (size_t i = 0; i < insts.size(); ++i) parse_instance(insts[i], "instances[" + std::to_string(i) + "]");
    const json& lights = root_.value("lights", json::array());
    for (size_t i = 0; i < lights.size(); ++i) parse_light(lights[i], "lights[" + std::to_string(i) + "]");
    if (needs_lut()) ensure_lut();
    sc.finalize();
    return scene_;
  }

 private:
  // ---------------------------------------------------------------- render / output / camera
  void parse_render(const json& r) {
    Scene& sc = *scene_;
    sc.render.spp = r.value("spp", sc.render.spp);
    sc.render.spp_per_pass = r.value("spp_per_pass", sc.render.spp_per_pass);
    sc.render.threads = r.value("threads", sc.render.threads);
    sc.render.backend = r.value("backend", sc.render.backend);
    sc.render.light_reference_distance = r.value("light_reference_distance", sc.render.light_reference_distance);
    sc.params.max_depth = r.value("max_depth", sc.params.max_depth);
    sc.params.rr_depth = r.value("rr_depth", sc.params.rr_depth);
    sc.params.seed = r.value("seed", uint64_t(0));
    sc.params.transparent_shadows = r.value("transparent_shadows", true) ? 1 : 0;
    sc.params.clamp_contribution = r.value("clamp", 0.f);
    std::string dm = r.value("depth_mode", std::string("distance"));
    if (dm != "distance" && dm != "z") fail("render.depth_mode", "must be 'distance' or 'z'");
    sc.params.depth_mode = dm == "z" ? kDepthZ : kDepthDistance;
  }

  void parse_output(const json& o) {
    OutputSettings& out = scene_->output;
    if (o.contains("exr")) out.exr = resolve_path(o["exr"], base_);
    if (o.contains("npz")) out.npz = resolve_path(o["npz"], base_);
    if (o.contains("preview_png")) out.preview_png = resolve_path(o["preview_png"], base_);
    out.exr_half = o.value("exr_half", false);
    if (o.contains("metadata")) out.metadata_json = o["metadata"].dump();
  }

  void parse_camera(const json& c) {
    CameraParams& cam = scene_->camera;
    if (c.value("type", std::string("pinhole")) != "pinhole")
      fail("camera.type", "only 'pinhole' is supported (lens effects: spectral_optics stage)");
    if (c.contains("resolution")) {
      cam.width = c["resolution"].at(0);
      cam.height = c["resolution"].at(1);
    }
    if (cam.width <= 0 || cam.height <= 0) fail("camera.resolution", "must be positive");
    double w = cam.width, h = cam.height;
    double tx = 0, ty = 0;
    if (c.contains("focal_length_mm")) {
      double f = c["focal_length_mm"], sw = c.value("sensor_width_mm", 36.0);
      double sh = c.value("sensor_height_mm", sw * h / w);
      tx = 0.5 * sw / f;
      ty = 0.5 * sh / f;
    } else {
      double fov = c.value("fov_deg", 60.0) * M_PI / 180.0;
      std::string axis = c.value("fov_axis", std::string("horizontal"));
      double t = std::tan(0.5 * fov);
      if (axis == "horizontal") {
        tx = t;
        ty = t * h / w;
      } else if (axis == "vertical") {
        ty = t;
        tx = t * w / h;
      } else if (axis == "diagonal") {
        double d = std::sqrt(w * w + h * h);
        tx = t * w / d;
        ty = t * h / d;
      } else {
        fail("camera.fov_axis", "horizontal | vertical | diagonal");
      }
    }
    cam.tan_half_fov_x = float(tx);
    cam.tan_half_fov_y = float(ty);
    if (c.contains("matrix")) {
      cam.cam_to_world = matrix4(c["matrix"], "camera.matrix");
      return;
    }
    Vec3 pos = c.contains("position") ? vec3(c["position"], "camera.position") : Vec3(0.f);
    Affine a = Affine::identity();
    if (c.contains("rotation")) {
      a = quat_rotation(c["rotation"], "camera.rotation");
    } else {
      Vec3 at = c.contains("look_at") ? vec3(c["look_at"], "camera.look_at") : pos + Vec3(0, 0, -1);
      Vec3 up = c.contains("up") ? vec3(c["up"], "camera.up") : Vec3(0, 1, 0);
      Vec3 fwd = normalize(at - pos);
      Vec3 right = cross(fwd, up);
      if (length(right) < 1e-6f) fail("camera", "look_at direction parallel to up");
      right = normalize(right);
      Vec3 u = cross(right, fwd);
      for (int r = 0; r < 3; ++r) {
        a.m[r][0] = right[r];
        a.m[r][1] = u[r];
        a.m[r][2] = -fwd[r];
      }
    }
    a.m[0][3] = pos.x;
    a.m[1][3] = pos.y;
    a.m[2][3] = pos.z;
    cam.cam_to_world = a;
  }

  // ---------------------------------------------------------------- spectra
  const CsvSpectra& csv(const std::string& path) {
    auto it = csv_cache_.find(path);
    if (it != csv_cache_.end()) return it->second;
    return csv_cache_.emplace(path, read_csv_spectra(path)).first->second;
  }

  std::string find_library_file(const std::string& name) {
    if (library_index_.empty()) {
      for (const auto& dir : library_dirs_) {
        std::error_code ec;
        if (!fs::exists(dir, ec)) continue;
        for (auto& e : fs::recursive_directory_iterator(dir, ec))
          if (e.is_regular_file() && e.path().extension() == ".csv")
            library_index_.emplace(e.path().stem().string(), e.path().string());
      }
    }
    auto it = library_index_.find(name);
    return it == library_index_.end() ? "" : it->second;
  }

  DenseSpectrum library_spectrum(const std::string& ref, const std::string& where) {
    std::string name = ref, col;
    auto colon = ref.find(':');
    if (colon != std::string::npos) {
      name = ref.substr(0, colon);
      col = ref.substr(colon + 1);
    }
    std::string file = find_library_file(name);
    if (file.empty()) {
      if (col.empty() && (ref == "D65" || ref == "A" || ref == "E")) return DenseSpectrum::cie_illuminant(ref);
      fail(where, "spectrum '" + ref + "' not found in library dirs");
    }
    const CsvSpectra& c = csv(file);
    int ci = col.empty() ? 0 : c.column_index(col);
    if (ci < 0) fail(where, "column '" + col + "' not in " + file);
    if (col.empty() && c.columns.size() > 1) fail(where, file + " has several columns; use '" + name + ":<column>'");
    return c.spectrum(ci);
  }

  enum class RgbMode { Reflectance, Illuminant };

  // Returns band values for a spectrum specification.
  std::vector<float> bands(const json& spec, const std::string& where, RgbMode rgb_mode) {
    const BandGrid& g = scene_->grid;
    if (spec.is_number()) return std::vector<float>(g.n, spec.get<float>());
    if (spec.is_string()) return library_spectrum(spec.get<std::string>(), where).to_bands(g);
    if (!spec.is_object()) fail(where, "spectrum must be a number, library name or object");
    std::string type = spec.value("type", std::string(spec.contains("value") ? "constant" : ""));
    std::vector<float> out;
    if (type == "constant") {
      out.assign(g.n, spec.at("value").get<float>());
    } else if (type == "library") {
      out = library_spectrum(spec.at("name"), where).to_bands(g);
    } else if (type == "csv") {
      const CsvSpectra& c = csv(resolve_path(spec.at("path"), base_));
      int ci = 0;
      if (spec.contains("column")) {
        ci = c.column_index(spec["column"].is_string() ? spec["column"].get<std::string>()
                                                         : std::to_string(spec["column"].get<int>()));
        if (ci < 0) fail(where, "column not found");
      }
      out = c.spectrum(ci).to_bands(g);
    } else if (type == "samples") {
      auto wl = spec.at("wavelengths").get<std::vector<float>>();
      auto v = spec.at("values").get<std::vector<float>>();
      Extrapolation ex = spec.value("extrapolation", std::string("clamp")) == "zero" ? Extrapolation::Zero
                                                                                    : Extrapolation::Clamp;
      out = DenseSpectrum::from_samples(wl, v, ex).to_bands(g);
    } else if (type == "blackbody") {
      out = DenseSpectrum::blackbody(spec.at("temperature")).to_bands(g);
    } else if (type == "cie") {
      out = DenseSpectrum::cie_illuminant(spec.at("name")).to_bands(g);
    } else if (type == "glass") {
      out = find_glass(spec.at("name")).spectrum().to_bands(g);
    } else if (type == "rgb") {
      auto rgb = spec.at("value").get<std::vector<float>>();
      if (rgb.size() != 3) fail(where, "rgb value must have 3 entries");
      ensure_lut();
      SceneView v = scene_->view();
      Spectrum s;
      if (rgb_mode == RgbMode::Reflectance) uplift_reflectance(v, rgb[0], rgb[1], rgb[2], s);
      else uplift_illuminant(v, rgb[0], rgb[1], rgb[2], s);
      out.assign(s.v, s.v + g.n);
    } else {
      fail(where, "unknown spectrum type '" + type + "'");
    }
    float scale = spec.value("scale", 1.f);
    if (spec.contains("photometric")) {
      // Scale so that 683 lm/W * integral(S * ybar) equals the given photometric value
      // (cd for intensities, cd/m^2 for radiances, lx for irradiances).
      DenseSpectrum d = from_bands(out, g);
      double y = double(d.xyz()[1]) * 683.0;
      if (!(y > 0)) fail(where, "photometric normalization of a spectrum with zero luminance");
      scale *= float(spec["photometric"].get<double>() / y);
    }
    std::string norm = spec.value("normalize", std::string());
    if (norm == "peak") {
      float m = *std::max_element(out.begin(), out.end());
      if (m > 0) scale /= m;
    } else if (!norm.empty()) {
      fail(where, "normalize must be 'peak'");
    }
    for (float& x : out) x *= scale;
    return out;
  }

  // True if any material relies on RGB uplifting of a non-gray colour.
  bool needs_lut() const {
    for (const auto& m : scene_->materials()) {
      bool gray = m.base_color[0] == m.base_color[1] && m.base_color[1] == m.base_color[2];
      if (m.type == kMatPbr && m.reflectance_spec < 0 && (m.base_color_tex >= 0 || !gray)) return true;
      bool egray = m.emissive[0] == m.emissive[1] && m.emissive[1] == m.emissive[2];
      if (m.emission_spec < 0 && (m.emissive_tex >= 0 || !egray)) return true;
    }
    return false;
  }

  void ensure_lut() {
    if (scene_->uplift) return;
    scene_->uplift = obtain_uplift_lut(lut_path_, lut_res_);
  }

  // ---------------------------------------------------------------- materials
  void parse_materials(const json& m) {
    library_dirs_.push_back((fs::path(data_dir()) / "spectra").string());
    for (const auto& d : m.value("library_dirs", json::array())) library_dirs_.push_back(resolve_path(d, base_));
    const json up = m.value("uplift", json::object());
    if (up.contains("lut")) lut_path_ = resolve_path(up["lut"], base_);
    lut_res_ = up.value("resolution", 64);
    scene_->params.uplift_hold_lambda = up.value("hold_lambda", 780.f);
    std::string method = up.value("method", std::string("jakob_hanika"));
    if (method != "jakob_hanika") fail("materials.uplift.method", "only 'jakob_hanika' is supported");
    for (const auto& o : m.value("overrides", json::array())) global_overrides_.push_back(o);
    for (const auto& [name, def] : m.value("definitions", json::object()).items()) {
      MaterialRecord rec;
      apply_material_fields(rec, def, "materials.definitions." + name);
      named_materials_[name] = scene_->add_material(rec, name);
    }
  }

  void apply_material_fields(MaterialRecord& m, const json& o, const std::string& where) {
    Scene& sc = *scene_;
    if (o.contains("bsdf")) {
      std::string b = o["bsdf"];
      if (b == "pbr") m.type = kMatPbr;
      else if (b == "dielectric") m.type = kMatDielectric;
      else if (b == "thin_dielectric") m.type = kMatThinDielectric;
      else fail(where + ".bsdf", "pbr | dielectric | thin_dielectric");
    }
    if (o.contains("base_color")) {
      auto c = o["base_color"].get<std::vector<float>>();
      for (size_t i = 0; i < c.size() && i < 4; ++i) m.base_color[i] = c[i];
      m.base_color_tex = o.value("keep_texture", false) ? m.base_color_tex : -1;
    }
    if (o.contains("reflectance"))
      m.reflectance_spec = sc.add_band_spectrum(bands(o["reflectance"], where + ".reflectance", RgbMode::Reflectance));
    if (o.contains("metallic")) m.metallic = o["metallic"];
    if (o.contains("roughness")) m.roughness = o["roughness"];
    if (o.contains("glass")) {
      GlassModel g = find_glass(o["glass"]);
      m.ior_spec = sc.add_band_spectrum(g.spectrum().to_bands(sc.grid));
      m.ior = float(g.ior(587.56));
    }
    if (o.contains("ior")) {
      if (o["ior"].is_number()) {
        m.ior = o["ior"];
        m.ior_spec = -1;
      } else {
        auto b = bands(o["ior"], where + ".ior", RgbMode::Reflectance);
        m.ior_spec = sc.add_band_spectrum(b);
        m.ior = from_bands(b, sc.grid).eval(587.56f);
      }
    }
    if (o.contains("transmittance"))
      m.transmittance_spec =
          sc.add_band_spectrum(bands(o["transmittance"], where + ".transmittance", RgbMode::Reflectance));
    if (o.contains("absorption"))
      m.absorption_spec = sc.add_band_spectrum(bands(o["absorption"], where + ".absorption", RgbMode::Reflectance));
    if (o.contains("internal_transmittance")) {
      // Measured internal transmittance T over thickness d -> absorption coefficient -ln(T)/d.
      double d = o.at("thickness_mm").get<double>() * 1e-3;
      if (!(d > 0)) fail(where + ".thickness_mm", "must be > 0");
      auto t = bands(o["internal_transmittance"], where + ".internal_transmittance", RgbMode::Reflectance);
      for (float& x : t) x = float(-std::log(std::max(1e-6f, std::min(1.f, x))) / d);
      m.absorption_spec = sc.add_band_spectrum(t);
    }
    if (o.contains("emission")) {
      m.emission_spec = sc.add_band_spectrum(bands(o["emission"], where + ".emission", RgbMode::Illuminant));
    }
    if (o.contains("emission_scale")) m.emission_scale = o["emission_scale"];
    if (o.contains("emissive")) {
      auto c = o["emissive"].get<std::vector<float>>();
      for (int i = 0; i < 3; ++i) m.emissive[i] = c.at(i);
    }
    if (o.contains("alpha_mode")) {
      std::string a = o["alpha_mode"];
      m.alpha_mode = a == "mask" ? kAlphaMask : (a == "blend" ? kAlphaBlend : kAlphaOpaque);
    }
    if (o.contains("alpha_cutoff")) m.alpha_cutoff = o["alpha_cutoff"];
    if (o.contains("double_sided")) m.double_sided = o["double_sided"].get<bool>() ? 1 : 0;
  }

  int32_t default_material() {
    if (default_material_ < 0) {
      MaterialRecord m;
      m.base_color[0] = m.base_color[1] = m.base_color[2] = 0.5f;
      default_material_ = scene_->add_material(m, "__default");
    }
    return default_material_;
  }

  int32_t resolve_material(const json& mj, const std::string& where) {
    if (mj.is_string()) {
      auto it = named_materials_.find(mj.get<std::string>());
      if (it == named_materials_.end()) fail(where, "unknown material '" + mj.get<std::string>() + "'");
      return it->second;
    }
    if (mj.is_object()) {
      MaterialRecord m;
      apply_material_fields(m, mj, where);
      return scene_->add_material(m, mj.value("name", where));
    }
    fail(where, "material must be a name or an object");
  }

  // ---------------------------------------------------------------- assets / instances
  const GltfAsset& asset(const std::string& key, const std::string& where) {
    auto it = assets_.find(key);
    if (it != assets_.end()) return it->second;
    auto d = asset_defs_.find(key);
    if (d == asset_defs_.end()) fail(where, "unknown asset '" + key + "'");
    std::string path = resolve_path(d->second.at("path"), base_);
    GltfAsset a = load_gltf(path, *scene_);
    // Global material overrides (applied in order, last wins).
    std::set<int32_t> done;
    for (auto& part : a.parts) {
      if (part.material < 0 || done.count(part.material)) continue;
      done.insert(part.material);
      for (const auto& ov : global_overrides_) {
        if (ov.contains("asset") && !regex_match_name(ov["asset"], key)) continue;
        if (!regex_match_name(ov.value("match", std::string(".*")), part.material_name)) continue;
        apply_material_fields(scene_->materials_mut()[part.material], ov, "materials.overrides");
      }
    }
    return assets_.emplace(key, std::move(a)).first->second;
  }

  void parse_instance(const json& o, const std::string& where) {
    Scene& sc = *scene_;
    Affine xf = parse_transform(o, where);
    uint32_t seg = o.value("seg_id", 0u);
    if (o.contains("asset")) {
      const GltfAsset& a = asset(o["asset"], where);
      std::map<int32_t, int32_t> cloned;
      const json ovs = o.value("material_overrides", json::array());
      const json seg_mat = o.value("seg_id_by_material", json::object());
      const json seg_node = o.value("seg_id_by_node", json::object());
      for (const auto& part : a.parts) {
        int32_t mat = part.material >= 0 ? part.material : default_material();
        const std::string& mname = sc.material_names()[mat];
        bool touched = false;
        for (const auto& ov : ovs)
          if (regex_match_name(ov.value("match", std::string(".*")), mname)) touched = true;
        if (touched) {
          auto it = cloned.find(mat);
          if (it == cloned.end()) {
            MaterialRecord m = sc.materials()[mat];
            for (const auto& ov : ovs)
              if (regex_match_name(ov.value("match", std::string(".*")), mname))
                apply_material_fields(m, ov, where + ".material_overrides");
            it = cloned.emplace(mat, sc.add_material(m, mname)).first;
          }
          mat = it->second;
        }
        uint32_t s = seg;
        for (const auto& [pat, id] : seg_mat.items())
          if (regex_match_name(pat, mname)) s = id.get<uint32_t>();
        for (const auto& [pat, id] : seg_node.items())
          if (regex_match_name(pat, part.node_name)) s = id.get<uint32_t>();
        sc.add_instance(part.mesh, compose(xf, part.node_to_asset), mat, s);
      }
      return;
    }
    if (o.contains("primitive")) {
      std::string p = o["primitive"];
      MeshData md;
      if (p == "sphere") md = make_sphere(o.value("radius", 1.f), o.value("segments", 64), o.value("rings", 32));
      else if (p == "quad") {
        auto s = o.value("size", std::vector<float>{1.f, 1.f});
        md = make_quad(s.at(0), s.at(1));
      } else if (p == "box") {
        auto s = o.value("size", std::vector<float>{1.f, 1.f, 1.f});
        md = make_box(s.at(0), s.at(1), s.at(2));
      } else if (p == "disk") {
        md = make_disk(o.value("radius", 1.f), o.value("segments", 64));
      } else {
        fail(where + ".primitive", "sphere | quad | box | disk");
      }
      if (o.value("flip_normals", false)) {
        for (auto& n : md.normals) n = -n;
        for (size_t t = 0; t < md.indices.size(); t += 3) std::swap(md.indices[t + 1], md.indices[t + 2]);
      }
      int32_t mat = o.contains("material") ? resolve_material(o["material"], where + ".material") : default_material();
      md.material = mat;
      uint32_t mesh = sc.add_mesh(md);
      sc.add_instance(mesh, xf, mat, seg);
      return;
    }
    fail(where, "instance needs 'asset' or 'primitive'");
  }

  // ---------------------------------------------------------------- lights
  void parse_light(const json& o, const std::string& where) {
    if (!o.value("enabled", true)) return;
    Scene& sc = *scene_;
    std::string type = o.value("type", std::string());
    LightRecord l;
    l.scale = o.value("scale", 1.f);
    if (type == "point" || type == "spot") {
      l.type = type == "point" ? kLightPoint : kLightSpot;
      l.position = vec3(o.at("position"), where + ".position");
      l.spectrum = sc.add_band_spectrum(bands(o.at("emission"), where + ".emission", RgbMode::Illuminant));
      if (type == "spot") {
        l.direction = normalize(vec3(o.at("direction"), where + ".direction"));
        float inner = o.value("cone_inner_deg", 20.f), outer = o.value("cone_outer_deg", 30.f);
        if (!(outer >= inner)) fail(where, "cone_outer_deg must be >= cone_inner_deg");
        l.cos_inner = std::cos(inner * kPi / 180.f);
        l.cos_outer = std::cos(outer * kPi / 180.f);
      }
      sc.add_light(l);
    } else if (type == "directional") {
      l.type = kLightDirectional;
      l.direction = normalize(vec3(o.at("direction"), where + ".direction"));
      l.spectrum = sc.add_band_spectrum(bands(o.at("emission"), where + ".emission", RgbMode::Illuminant));
      sc.add_light(l);
    } else if (type == "sky") {
      parse_sky(o, where);
    } else if (type == "envmap") {
      parse_envmap(o, where);
    } else {
      fail(where + ".type", "point | spot | directional | sky | envmap (area lights: material emission)");
    }
  }

  SkyParams sky_params(const json& o, const std::string& where) {
    SkyParams p;
    const json sun = o.value("sun", json::object());
    if (sun.contains("direction")) p.sun_direction = normalize(vec3(sun["direction"], where + ".sun.direction"));
    else p.sun_direction = sun_direction_from_angles(sun.value("elevation_deg", 45.0), sun.value("azimuth_deg", 180.0));
    p.visibility_km = o.value("visibility_km", 59.4);
    p.ground_albedo = o.value("ground_albedo", 0.33);
    p.altitude_m = o.value("altitude_m", 0.0);
    return p;
  }

  void parse_sky(const json& o, const std::string& where) {
    Scene& sc = *scene_;
    std::string model = o.value("model", std::string("simple"));
    SkyParams p = sky_params(o, where);
    SkyTabulation tab;
    tab.scale = o.value("scale", 1.f);
    std::unique_ptr<SkyModel> sky;
    if (model == "simple") {
      DenseSpectrum skyL = o.contains("sky_radiance")
                               ? from_bands(bands(o["sky_radiance"], where + ".sky_radiance", RgbMode::Illuminant), sc.grid)
                               : DenseSpectrum::constant(0.f);
      double half = o.value("sun", json::object()).value("half_angle_deg", 0.2667) * M_PI / 180.0;
      DenseSpectrum sunL = DenseSpectrum::constant(0.f);
      const json sun = o.value("sun", json::object());
      if (sun.contains("radiance")) {
        sunL = from_bands(bands(sun["radiance"], where + ".sun.radiance", RgbMode::Illuminant), sc.grid);
      } else if (sun.contains("irradiance")) {
        // Normal irradiance E -> uniform disk radiance L = E / solid angle.
        double omega = 2.0 * M_PI * (1.0 - std::cos(half));
        sunL = from_bands(bands(sun["irradiance"], where + ".sun.irradiance", RgbMode::Illuminant), sc.grid) *
               float(1.0 / omega);
      }
      sky = make_simple_sky(skyL, sunL, half);
      tab.width = 32;
      tab.height = 16;
      tab.supersample = 1;
    } else if (model == "prague") {
      std::string ds = resolve_path(o.at("dataset"), base_);
      sky = make_prague_sky(ds, o.value("load_single_visibility", false) ? p.visibility_km : 0.0);
    } else if (model == "plugin") {
      sky = load_sky_plugin(resolve_path(o.at("library"), base_), o.value("options", json::object()).dump());
    } else {
      fail(where + ".model", "simple | prague | plugin");
    }
    if (o.contains("table_resolution")) {
      tab.width = o["table_resolution"].at(0);
      tab.height = o["table_resolution"].at(1);
    } else if (model != "simple") {
      tab.width = 512;
      tab.height = 256;
    }
    sky->configure(p);
    if (model != "simple" && o.value("cache", true)) {
      json key = o;
      key["_grid"] = {sc.grid.lambda_min, sc.grid.step, sc.grid.n, tab.width, tab.height};
      size_t h = std::hash<std::string>{}(key.dump());
      tab.cache_path = (fs::path(cache_dir()) / ("sky_" + std::to_string(h) + ".bin")).string();
    }
    add_sky_to_scene(*sky, p, tab, sc);
  }

  void parse_envmap(const json& o, const std::string& where) {
    Scene& sc = *scene_;
    RgbImage img = load_rgb_image(resolve_path(o.at("path"), base_));
    int W = 512, H = 256;
    if (o.contains("resolution")) {
      W = o["resolution"].at(0);
      H = o["resolution"].at(1);
    }
    W = std::min(W, img.width);
    H = std::min(H, img.height);
    ensure_lut();
    SceneView v = sc.view();
    std::vector<float> table(size_t(W) * H * sc.grid.n);
    // Box-filter downsample in RGB, then uplift each texel as an illuminant.
    for (int y = 0; y < H; ++y) {
      int y0 = y * img.height / H, y1 = std::max(y0 + 1, (y + 1) * img.height / H);
      for (int x = 0; x < W; ++x) {
        int x0 = x * img.width / W, x1 = std::max(x0 + 1, (x + 1) * img.width / W);
        double c[3] = {0, 0, 0};
        for (int yy = y0; yy < y1; ++yy)
          for (int xx = x0; xx < x1; ++xx)
            for (int k = 0; k < 3; ++k) c[k] += img.rgb[(size_t(yy) * img.width + xx) * 3 + k];
        double n = double(y1 - y0) * (x1 - x0);
        Spectrum s;
        uplift_illuminant(v, float(c[0] / n), float(c[1] / n), float(c[2] / n), s);
        std::copy(s.v, s.v + sc.grid.n, table.begin() + (size_t(y) * W + x) * sc.grid.n);
      }
    }
    (void)where;
    sc.set_environment(W, H, std::move(table), o.value("scale", 1.f), o.value("rotation_deg", 0.f) * kPi / 180.f);
  }

  const json& root_;
  std::string base_;
  std::shared_ptr<Scene> scene_;
  std::vector<std::string> library_dirs_;
  std::map<std::string, std::string> library_index_;
  std::map<std::string, CsvSpectra> csv_cache_;
  std::map<std::string, int32_t> named_materials_;
  std::map<std::string, json> asset_defs_;
  std::map<std::string, GltfAsset> assets_;
  std::vector<json> global_overrides_;
  std::string lut_path_;
  int lut_res_ = 64;
  int32_t default_material_ = -1;
};

}  // namespace

std::shared_ptr<Scene> load_scene_json(const json& j, const std::string& base_dir) {
  Loader l(j, base_dir);
  return l.load();
}

std::shared_ptr<Scene> load_scene_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Cannot open scene file " + path);
  json j;
  try {
    j = json::parse(in, nullptr, true, true);  // allow comments
  } catch (const json::parse_error& e) {
    throw std::runtime_error("Invalid JSON in " + path + ": " + e.what());
  }
  std::string base = fs::absolute(fs::path(path)).parent_path().string();
  return load_scene_json(j, base);
}

}  // namespace spectral
