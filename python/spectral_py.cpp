// Python bindings of the live rendering Session (module `_spectral`, wrapped by the
// `spectral_renderer` package in python/spectral_renderer).
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>

#include "spectral/io/image_io.h"
#include "spectral/session.h"

namespace py = pybind11;
using namespace spectral;

namespace {

Affine to_affine(py::array_t<double, py::array::c_style | py::array::forcecast> m) {
  if (m.ndim() != 2 || m.shape(1) != 4 || (m.shape(0) != 3 && m.shape(0) != 4))
    throw std::invalid_argument("expected a 3x4 or 4x4 matrix");
  Affine a;
  auto r = m.unchecked<2>();
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 4; ++j) a.m[i][j] = float(r(i, j));
  return a;
}

template <class T>
py::array_t<T> to_array(const std::vector<T>& v, std::vector<py::ssize_t> shape) {
  py::array_t<T> a(shape);
  std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
  return a;
}

}  // namespace

PYBIND11_MODULE(_spectral, m) {
  m.doc() = "Dense-band spectral path tracer: live rendering session";
  m.attr("__version__") = Renderer::version();

  py::class_<Session>(m, "Session")
      .def(py::init<const std::string&, int>(), py::arg("backend") = "auto", py::arg("threads") = 0)
      .def("load_file", &Session::load_file, py::arg("path"))
      .def("load_json", &Session::load_json, py::arg("text"), py::arg("base_dir") = ".")
      .def_property_readonly("backend", &Session::backend_name)
      .def_property_readonly("width", [](Session& s) { return s.scene().camera.width; })
      .def_property_readonly("height", [](Session& s) { return s.scene().camera.height; })
      .def_property_readonly("wavelengths",
                             [](Session& s) {
                               const BandGrid& g = s.scene().grid;
                               std::vector<float> w(g.n);
                               for (int i = 0; i < g.n; ++i) w[i] = g.center(i);
                               return to_array(w, {g.n});
                             })
      .def_property_readonly("spp", &Session::spp)
      .def("live_objects", &Session::live_objects)
      .def("add_asset", &Session::add_asset, py::arg("key"), py::arg("path"))
      .def("spawn", &Session::spawn, py::arg("instance_json"))
      .def("set_transform", [](Session& s, int h, py::array m) { s.set_transform(h, to_affine(m)); },
           py::arg("handle"), py::arg("matrix"))
      .def("set_seg_id", &Session::set_seg_id)
      .def("remove", &Session::remove, py::arg("handle"))
      .def("set_camera",
           [](Session& s, py::array m, float fov, int w, int h) { s.set_camera(to_affine(m), fov, w, h); },
           py::arg("cam_to_world"), py::arg("fov_x_deg"), py::arg("width"), py::arg("height"))
      .def("set_camera_json", &Session::set_camera_json)
      .def("set_dynamic_lights", &Session::set_dynamic_lights, py::arg("lights_json"))
      .def("set_sun", [](Session& s, float x, float y, float z) { s.set_sun(Vec3(x, y, z)); })
      .def("reset", &Session::reset)
      .def("render", &Session::render, py::arg("spp"), py::call_guard<py::gil_scoped_release>())
      .def(
          "preview",
          [](Session& s, const std::string& mode, float band_nm, std::vector<float> rgb_nm, bool auto_exposure,
             float exposure_ev, float depth_max) {
            PreviewOptions o;
            o.mode = mode;
            o.band_nm = band_nm;
            if (rgb_nm.size() != 3) throw std::invalid_argument("rgb_nm needs 3 wavelengths");
            for (int k = 0; k < 3; ++k) o.rgb_nm[k] = rgb_nm[k];
            o.auto_exposure = auto_exposure;
            o.exposure_ev = exposure_ev;
            o.depth_max = depth_max;
            std::vector<uint8_t> rgb;
            {
              py::gil_scoped_release release;
              rgb = s.preview(o);
            }
            return to_array(rgb, {s.scene().camera.height, s.scene().camera.width, 3});
          },
          py::arg("mode") = "srgb", py::arg("band_nm") = 550.f,
          py::arg("rgb_nm") = std::vector<float>{850.f, 650.f, 550.f}, py::arg("auto_exposure") = true,
          py::arg("exposure_ev") = 0.f, py::arg("depth_max") = 100.f)
      .def("image",
           [](Session& s) {
             SpectralImage img = s.image();
             py::dict d;
             d["radiance"] = to_array(img.radiance, {img.height, img.width, img.grid.n});
             d["wavelengths"] = to_array(img.wavelengths(), {img.grid.n});
             d["depth"] = to_array(img.depth, {img.height, img.width});
             d["seg_id"] = to_array(img.seg_id, {img.height, img.width});
             d["metadata"] = img.metadata_json;
             d["spp"] = img.spp;
             return d;
           })
      .def(
          "save",
          [](Session& s, const std::string& npz, const std::string& exr, const std::string& png) {
            SpectralImage img = s.image();
            OutputSettings out;
            out.npz = npz;
            out.exr = exr;
            out.preview_png = png;
            Renderer::write_outputs(img, out);
          },
          py::arg("npz") = "", py::arg("exr") = "", py::arg("preview_png") = "");
}
