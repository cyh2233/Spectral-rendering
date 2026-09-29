# CARLA → spectral renderer (offline)

1. **Export assets once** (Unreal Editor with the CARLA project): export the town map and the
   vehicle/walker blueprints you need to glTF (`.glb`) with Unreal's glTF Exporter plugin
   (File › Export, format glTF). Keep material names (they drive `materials_default.json`
   regexes). Tips: export the town per semantic class (roads, buildings, vegetation) when possible
   so segmentation IDs are per instance; foliage cards need `alpha_mode: mask`.
   Alternatively import the FBX exports into Blender, clean them up, and export with the Blender
   add-on.
2. **Describe the mapping** from CARLA map/blueprint names to glTF files:
   `tools/carla_export/asset_mapping.example.json` (town, blueprints, segmentation rules,
   material overrides, headlight parameters, render settings).
3. **Per frame**: with CARLA running and a camera sensor attached,
   ```bash
   python tools/carla_export/carla_to_scene.py --mapping mapping.json --out frames/f000123.json \
          --camera-role hero_camera --headlights --dump-snapshot frames/f000123.snapshot.json
   spectral_render frames/f000123.json
   ```
   `--snapshot` re-creates scenes from saved snapshots without CARLA.

Frames: CARLA/Unreal is left-handed (X forward, Y right, Z up); the renderer is right-handed
Y-up. The converter maps (x, y, z)_UE → (y, z, −x) and transforms M = S·M_UE·Sᵀ; assets exported
by Unreal's glTF exporter are already in glTF axes (use a `fix` matrix per town entry otherwise).
Camera: horizontal FOV and image size from the sensor attributes. Sun: from the weather's
altitude/azimuth (check the azimuth convention of your CARLA version; override with
`--sun-direction`). Segmentation: CARLA `semantic_tags` of each actor (CityObjectLabel ids).
Headlights: two spot lights at the bounding-box front corners for vehicles with low/high beams
(CIE LED-B5 spectrum, 8 000 / 20 000 cd by default).

Tests: `python -m pytest tools/carla_export` (no CARLA needed).
