# Data sources

- **mnist_subset.npz**: a balanced subset of the MNIST database of handwritten digits (Y. LeCun, C. Cortes and
  C. J. C. Burges, http://yann.lecun.com/exdb/mnist/), downloaded from the mirror
  https://ossci-datasets.s3.amazonaws.com/mnist/ (the four `*-idx?-ubyte.gz` files, MD5 checksums verified:
  train-images f68b3c2dcbeaaa9fbdd348bbdeb94873, train-labels d53e105ee54ea40749a09fcbcd1e9432,
  t10k-images 9fb629c4189551a2d022fa330f9573f3, t10k-labels ec29112dd5afa0611ce80d1b7f02629c).
  Licence: Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0). The share-alike condition applies to this
  bundled subset: it is redistributed under the same licence.
  Content: arrays `images` (3000, 28, 28) uint8 and `labels` (3000,) uint8. Rows 0–999 are the first 100 images of
  every digit in the MNIST training file (file order), the pool from which the small training sets are drawn; rows
  1000–2999 are the first 200 images of every digit in the MNIST test file, the fixed test set. Made by
  `tensors.build_subset(raw_dir)`, which reads the raw IDX files with gzip and numpy only. At run time the images are
  averaged over 2×2 blocks to 14×14.
