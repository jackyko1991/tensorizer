# Data sources

- **mnist_train.npz, mnist_test.npz**: a balanced subset of the MNIST database of handwritten digits, Yann LeCun, Corinna Cortes and
  Christopher J. C. Burges (http://yann.lecun.com/exdb/mnist/; see also Y. LeCun, L. Bottou, Y. Bengio and P. Haffner,
  "Gradient-based learning applied to document recognition", Proc. IEEE 86(11), 2278–2324, 1998,
  doi:10.1109/5.726791). Downloaded from the PyTorch mirror https://ossci-datasets.s3.amazonaws.com/mnist/
  (files train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz, t10k-images-idx3-ubyte.gz,
  t10k-labels-idx1-ubyte.gz; MD5 f68b3c2dcbeaaa9fbdd348bbdeb94873, d53e105ee54ea40749a09fcbcd1e9432,
  9fb629c4189551a2d022fa330f9573f3, ec29112dd5afa0611ce80d1b7f02629c, the published checksums).
  **Licence: Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0)**, as stated by the dataset authors.
  The share-alike condition applies to this bundled subset: it is redistributed under CC BY-SA 3.0, with the
  attribution above.
  Processing (read with plain gzip + numpy from the idx headers, nothing executed): the first 300 images of each
  digit in the training file's order (3000 images) and the first
  100 of each digit in the test file's order (1000 images); each file holds `images` (N, 28, 28) uint8, `labels`
  (N,) uint8 and `index` (N,), the position in the original file; saved with numpy.savez_compressed (~0.7 MB in all).
- **Snakefile**: written for this problem (no external source): the workflow that the tensor engine parses. It is
  never run by Snakemake here.
