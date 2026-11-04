# Data sources

- `spot_vertices.csv`, `spot_faces.csv`: "Spot" (the cow) by Keenan Crane,
  https://www.cs.cmu.edu/~kmcrane/Projects/ModelRepository/ (file `spot.zip`, `spot_triangulated.obj`).
  Licence: CC0 1.0 Universal (public domain dedication; the archive's README says "As the sole author of this data,
  I hereby release it into the public domain"). Processing: none beyond format conversion. The 2930 vertex positions
  (x, y, z, 6 significant digits) and the 5856 triangles (0-based vertex indices) were copied from the OBJ file; the
  texture coordinates were dropped. Suggested citation: K. Crane, U. Pinkall and P. Schröder, "Robust fairing via
  conformal curvature flow", ACM Transactions on Graphics 32(4), 2013, doi:10.1145/2461912.2461986.
