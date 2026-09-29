"""Command line: build optics packages and apply them to renderer outputs.

  spectral-optics windshield --focal-length 6 --f-number 1.8 --fov 60 40 -o ws.optics.npz [--rake 60 ...]
  spectral-optics vendor manifest.json -o lens.optics.npz
  spectral-optics prescription lens.zmx --fields 0 10 20 30 -o lens.optics.npz
  spectral-optics combine lens.optics.npz ws.optics.npz -o camera.optics.npz
  spectral-optics apply render.npz camera.optics.npz -o optical_image.npz [--irradiance]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from .package import OpticsPackage


def _fields_grid(fov_x, fov_y, n):
    xs = np.linspace(-fov_x / 2, fov_x / 2, n)
    ys = np.linspace(-fov_y / 2, fov_y / 2, n)
    return [(x, y) for y in ys[::-1] for x in xs]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="spectral-optics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("windshield", help="ideal lens behind a windshield")
    w.add_argument("--focal-length", type=float, required=True)
    w.add_argument("--f-number", type=float, required=True)
    w.add_argument("--fov", type=float, nargs=2, required=True, metavar=("X_DEG", "Y_DEG"))
    w.add_argument("--grid", type=int, default=3)
    w.add_argument("--rake", type=float, default=60.0)
    w.add_argument("--radius", type=float, default=4000.0)
    w.add_argument("--distance", type=float, default=60.0)
    w.add_argument("--wedge-mrad", type=float, default=0.0)
    w.add_argument("--layers", default="soda_lime:2.1,pvb_approx:0.76,soda_lime:2.1")
    w.add_argument("--wavelengths", type=float, nargs="+", default=[400, 450, 500, 550, 600, 650, 700, 800, 900, 1000])
    w.add_argument("--psf-pixel-um", type=float, default=0.5)
    w.add_argument("--psf-size", type=int, default=64)
    w.add_argument("-o", "--output", required=True)

    v = sub.add_parser("vendor", help="vendor data manifest -> package")
    v.add_argument("manifest")
    v.add_argument("-o", "--output", required=True)

    p = sub.add_parser("prescription", help=".zmx/.seq lens via RayOptics")
    p.add_argument("path")
    p.add_argument("--fields", type=float, nargs="+", required=True, help="field angles along +y (deg)")
    p.add_argument("--wavelengths", type=float, nargs="+", default=[450, 550, 650, 850])
    p.add_argument("--grid", type=int, default=5, help="expand to an NxN field grid")
    p.add_argument("-o", "--output", required=True)

    c = sub.add_parser("combine", help="convolve the PSFs of a lens package with a windshield package")
    c.add_argument("lens")
    c.add_argument("windshield")
    c.add_argument("-o", "--output", required=True)

    a = sub.add_parser("apply", help="apply a package to a renderer NPZ")
    a.add_argument("render")
    a.add_argument("package")
    a.add_argument("-o", "--output", required=True)
    a.add_argument("--irradiance", action="store_true", help="convert radiance to sensor irradiance")

    args = ap.parse_args(argv)
    if args.cmd == "windshield":
        from .windshield import Windshield, windshield_package
        layers = [(s.split(":")[0], float(s.split(":")[1])) for s in args.layers.split(",")]
        ws = Windshield(layers=layers, rake_deg=args.rake, radius_mm=args.radius, distance_mm=args.distance,
                        wedge_mrad=args.wedge_mrad)
        pkg = windshield_package(ws, args.focal_length, args.f_number, args.wavelengths,
                                 _fields_grid(args.fov[0], args.fov[1], args.grid), args.psf_pixel_um, args.psf_size)
    elif args.cmd == "vendor":
        from .vendor import build_package_from_manifest
        pkg = build_package_from_manifest(args.manifest)
    elif args.cmd == "prescription":
        from .lens_rayoptics import lens_package, load_prescription
        from .vendor import expand_rotational
        pkg = lens_package(load_prescription(args.path), args.wavelengths, args.fields)
        pkg = expand_rotational(pkg, max(abs(f) for f in args.fields), args.grid)
    elif args.cmd == "combine":
        from .combine import combine_packages
        pkg = combine_packages(OpticsPackage.load(args.lens), OpticsPackage.load(args.windshield))
    elif args.cmd == "apply":
        from .apply import apply_to_npz
        apply_to_npz(args.render, OpticsPackage.load(args.package), args.output, args.irradiance)
        print(f"wrote {args.output}")
        return 0
    pkg.save(args.output)
    print(f"wrote {args.output}: {len(pkg.wavelengths_nm)} wavelengths, {len(pkg.fields_deg)} fields, "
          f"PSF {pkg.psf.shape[-1]}x{pkg.psf.shape[-1]} @ {pkg.psf_pixel_um} um")
    return 0


if __name__ == "__main__":
    sys.exit(main())
