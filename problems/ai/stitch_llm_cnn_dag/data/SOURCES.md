# Data sources

## mnist_subset.npz (MNIST, CC BY-SA 3.0)

- **What**: 3000 training and 1000 test images of handwritten digits, 28×28 uint8, with their labels: a balanced
  subset (300 per digit from the MNIST training set, 100 per digit from the MNIST test set), about 0.7 MB.
- **Source**: the MNIST database of handwritten digits, Yann LeCun, Corinna Cortes and Christopher J. C. Burges.
  Original files `train-images-idx3-ubyte.gz`, `train-labels-idx1-ubyte.gz`, `t10k-images-idx3-ubyte.gz`,
  `t10k-labels-idx1-ubyte.gz`, downloaded from the PyTorch mirror https://ossci-datasets.s3.amazonaws.com/mnist/
  and checked against the published MD5 sums.
- **Licence**: Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0), as stated by the authors. This subset is
  a derivative and is shared under the same licence.
- **Citation**: Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document
  recognition", Proc. IEEE 86(11), 2278–2324 (1998).
- **Processing**: `data/make_subset.py` (numpy and gzip only; IDX headers checked): for each digit, draw 300
  training and 100 test indices without replacement with `numpy.random.default_rng(0)`, shuffle, and store
  `images` (4000, 28, 28) uint8, `labels` (4000,) uint8, `split` (0 = training, 1 = test) and `source_index` (the
  index in the original MNIST file).

## Sentences (generated, no data file)

The text is generated in code (`tensors.sentence`): "<a> <op> <b> is <r>." for every digit word a, b, every op in
plus, minus, times, and r = a op b (mod 10), 300 sentences in all, with the first word right-aligned in a 5-character
slot ("  two plus three is five."). No text is downloaded.

## training.smk

The training process as a Snakefile-style rule graph (written for this problem). It is parsed by the tensor DAG
engine and never run by Snakemake; its input/output names are labels, no such files are written.
