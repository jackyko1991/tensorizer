# Data sources

- **mnist_subset.npz**: a balanced subset of the MNIST database of handwritten digits (Y. LeCun, C. Cortes and
  C. J. C. Burges, http://yann.lecun.com/exdb/mnist/), downloaded from the mirror
  https://ossci-datasets.s3.amazonaws.com/mnist/ (`train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz`,
  MD5 checksums verified: f68b3c2dcbeaaa9fbdd348bbdeb94873 and d53e105ee54ea40749a09fcbcd1e9432).
  Licence: Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0). The share-alike condition applies to this
  bundled subset: it is redistributed under the same licence.
  Content: arrays `images` (2500, 28, 28) uint8 and `labels` (2500,) uint8, the first 250 images of every digit in
  the MNIST training file, in file order. Made by `tensors.build_subset(raw_dir)`, which reads the raw IDX files
  with gzip and numpy only.

- **umap_learn_reference.npz** and **umap_learn_reference.csv**: generated (computed offline, once) by
  `make_reference.py` in a throwaway environment outside the repository
  (`pixi exec --spec python=3.12 --spec umap-learn --spec scikit-learn python -I make_reference.py ...`), because
  umap-learn and scikit-learn are not dependencies of this project. umap-learn 0.5.12 (BSD-3-Clause,
  https://github.com/lmcinnes/umap), scikit-learn 1.9.1 (BSD-3-Clause), numpy 2.5.4, Python 3.12, Windows.
  The npz holds the 2D layouts by `umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)` of the MNIST subset
  above (`mnist`, (2500, 2) float32) and of the Swiss roll of `tensors.swiss_roll_numpy` (`swiss_roll`,
  (1500, 2)), scikit-learn's PCA layouts (`mnist_pca`, `swiss_roll_pca`) and a checksum of the Swiss-roll points
  (`swiss_checksum`). The CSV holds the versions, umap-learn's fitted `a`, `b`, the wall times of two calls of
  `fit_transform` (the first includes numba compilation) and scikit-learn's trustworthiness and 10-fold 10-NN
  accuracy, used by the tests to cross-check the tensor metrics. Licence: as the problem; the MNIST layout is
  derived from the CC BY-SA 3.0 subset and shares its licence.

- **make_reference.py**: the script that computed the reference above (not run by the problem; source code of
  this project, licence as the problem).

The Swiss roll itself is synthetic, generated in code (no file).
