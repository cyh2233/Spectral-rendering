#!/usr/bin/env python3
"""Renders the same scene with the CPU and CUDA backends and compares the spectral cubes.

Usage: python scripts/compare_backends.py build/spectral_render scene.json [--spp 256]
Expect agreement within Monte Carlo noise (the two backends use identical kernels and random
sequences; results differ only by floating-point details such as FMA contraction).
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np


def render(exe, scene, backend, spp, out):
    cmd = [exe, scene, "--backend", backend, "--spp", str(spp), "--out-npz", out, "--quiet"]
    subprocess.run(cmd, check=True)
    return np.load(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("renderer")
    ap.add_argument("scene")
    ap.add_argument("--spp", type=int, default=256)
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as d:
        cpu = render(a.renderer, a.scene, "cpu", a.spp, os.path.join(d, "cpu.npz"))
        gpu = render(a.renderer, a.scene, "cuda", a.spp, os.path.join(d, "gpu.npz"))
    rc, rg = cpu["radiance"].astype(np.float64), gpu["radiance"].astype(np.float64)
    band_mean_c, band_mean_g = rc.mean(axis=(0, 1)), rg.mean(axis=(0, 1))
    rel = np.abs(band_mean_g - band_mean_c) / np.maximum(np.abs(band_mean_c), 1e-12)
    print(f"image-mean per-band relative difference: max {rel.max():.4%}, mean {rel.mean():.4%}")
    print(f"pixel RMS difference / mean: {np.sqrt(((rg - rc) ** 2).mean()) / max(rc.mean(), 1e-12):.4%}")
    print(f"depth identical: {np.allclose(cpu['depth'], gpu['depth'], rtol=1e-4)}; "
          f"seg_id identical: {np.array_equal(cpu['seg_id'], gpu['seg_id'])}")
    sys.exit(0 if rel.mean() < 0.01 else 1)


if __name__ == "__main__":
    main()
