# Material spectra

Files named `approx_*.csv` are **hand-shaped placeholder curves**, not measurements. They exist so
that the material-assignment pipeline (regex overrides in `data/materials_default.json`) can be
exercised end to end, and they reproduce qualitative features only (vegetation red edge, Fe2+
absorption of green windshield glass, NIR cut of IR-reflective coatings).

Replace them with measured data before any quantitative use, for example:
- USGS Spectral Library Version 7 (splib07) -- asphalt, concrete, vegetation, paints (350-2500 nm).
- ECOSTRESS / ASTER spectral library -- man-made materials, vegetation, soils.
- Supplier data sheets -- windshield glass transmittance, retroreflective sheeting, automotive paint.

CSV format: first column wavelength in nm, one or more value columns, `#` comments,
optional `# extrapolation: clamp|zero` line, optional header row. Any `*.csv` below a
library directory can be referenced by its file stem (`"reflectance": "approx_concrete"`),
multi-column files as `"file:column"`.
