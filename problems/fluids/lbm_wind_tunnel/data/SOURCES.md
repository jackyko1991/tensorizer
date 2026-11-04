# Data sources

- `naca0012_wing.stl`: generated for this project by a short NumPy script (lofted sections with fan end caps) from the NACA four-digit thickness formula
  (NACA 0012, closed trailing edge), chord 1000 mm, span 1200 mm, extruded along y; 312 triangles, closed, binary STL.
  Licence: generated for this project (CC0 / public domain).
- `airplane.stl`: generated for this project by the same NumPy script: a chunky model plane (no real aircraft): fuselage of
  revolution (length 1000 mm, radius 70 mm, upswept tail cone), swept tapered main wing (NACA 0018 section, span 900 mm,
  25° sweep, chord 450 → 260 mm, 4° dihedral), horizontal and vertical tail (NACA 0018). Each part is a closed,
  outward-oriented surface; the parts overlap where they join (3076 triangles, binary STL). Licence: generated for this
  project (CC0 / public domain).
- `openlb_reference_sphere_force.csv`, `openlb_reference_sphere_midplane.csv`: output of one offline run of OpenLB 1.7
  (see the `#` header lines in each file for the version, driver and parameters). Licence: numbers generated for this
  project (CC0); OpenLB itself (GPL-2.0) is not included.
