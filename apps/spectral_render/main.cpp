// spectral_render: command-line front end of libspectral.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "spectral/api.h"

static void usage() {
  std::printf(
      "usage: spectral_render scene.json [options]\n"
      "  --spp N            samples per pixel\n"
      "  --backend B        cpu | cuda | auto\n"
      "  --threads N        CPU threads (0 = all)\n"
      "  --seed S           random seed\n"
      "  --resolution WxH   override image size\n"
      "  --out-exr PATH     multispectral EXR output\n"
      "  --out-npz PATH     NumPy NPZ output\n"
      "  --preview PATH     sRGB preview PNG\n"
      "  --quiet            no progress output\n");
}

int main(int argc, char** argv) {
  if (argc < 2 || !std::strcmp(argv[1], "-h") || !std::strcmp(argv[1], "--help")) {
    usage();
    return argc < 2 ? 1 : 0;
  }
  std::string scene_path;
  spectral::RenderOptions opt;
  std::string exr, npz, png;
  bool quiet = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", a.c_str());
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--spp") opt.spp = std::stoi(next());
    else if (a == "--backend") opt.backend = next();
    else if (a == "--threads") opt.threads = std::stoi(next());
    else if (a == "--seed") opt.seed = std::stoll(next());
    else if (a == "--resolution") {
      std::string r = next();
      auto x = r.find('x');
      if (x == std::string::npos) {
        std::fprintf(stderr, "resolution must be WxH\n");
        return 1;
      }
      opt.width = std::stoi(r.substr(0, x));
      opt.height = std::stoi(r.substr(x + 1));
    } else if (a == "--out-exr") exr = next();
    else if (a == "--out-npz") npz = next();
    else if (a == "--preview") png = next();
    else if (a == "--quiet") quiet = true;
    else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "unknown option %s\n", a.c_str());
      usage();
      return 1;
    } else scene_path = a;
  }
  try {
    auto t0 = std::chrono::steady_clock::now();
    spectral::Renderer r;
    r.load_scene_file(scene_path);
    r.set_options(opt);
    if (!quiet)
      r.set_progress_callback([](int done, int total) {
        std::fprintf(stderr, "\r[spectral] %d / %d spp", done, total);
        std::fflush(stderr);
      });
    auto t1 = std::chrono::steady_clock::now();
    spectral::SpectralImage img = r.render();
    auto t2 = std::chrono::steady_clock::now();
    spectral::OutputSettings out = r.scene().output;
    if (!exr.empty()) out.exr = exr;
    if (!npz.empty()) out.npz = npz;
    if (!png.empty()) out.preview_png = png;
    spectral::Renderer::write_outputs(img, out);
    if (!quiet) {
      std::fprintf(stderr, "\n[spectral] %dx%d, %d bands (%.1f-%.1f nm), %d spp; load %.2fs, render %.2fs\n", img.width,
                   img.height, img.grid.n, img.grid.lambda_min, img.grid.lambda_max(), img.spp,
                   std::chrono::duration<double>(t1 - t0).count(), std::chrono::duration<double>(t2 - t1).count());
      if (!out.exr.empty()) std::fprintf(stderr, "[spectral] wrote %s\n", out.exr.c_str());
      if (!out.npz.empty()) std::fprintf(stderr, "[spectral] wrote %s\n", out.npz.c_str());
      if (!out.preview_png.empty()) std::fprintf(stderr, "[spectral] wrote %s\n", out.preview_png.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
  return 0;
}
