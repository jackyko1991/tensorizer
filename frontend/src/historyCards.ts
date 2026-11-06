/** Cards of the Concept "Tensor history" tab (#/concept/history[/<card id>]). Text lives in content/pages.<lang>.json
 *  under history.<id>.{title,alt,p1,p2}; pictures are Wikimedia Commons files (public domain, CC0, CC BY, CC BY-SA)
 *  resized into public/history/ (credits in public/history/CREDITS.md), or a drawn SVG where no free photo exists. */

export type Part = 1 | 2 | 3;
export type Drawing = "im2col" | "dataflow" | "tape" | "tensorcore" | "einsum";

export interface Photo {
  file: string;
  author: string;
  licence: string;
  licenceUrl?: string;
  /** Commons file page */
  source: string;
}

export interface HistoryCard {
  id: string;
  part: Part;
  /** shown on the timeline; null = "Today" (translated) */
  year: string | null;
  photo?: Photo;
  drawing?: Drawing;
  sources: { label: string; url: string }[];
}

const PD = "Public domain";
const CC0: Pick<Photo, "licence" | "licenceUrl"> = {
  licence: "CC0",
  licenceUrl: "https://creativecommons.org/publicdomain/zero/1.0/",
};
const commons = (name: string) => `https://commons.wikimedia.org/wiki/File:${encodeURIComponent(name.replace(/ /g, "_"))}`;

export const HISTORY_CARDS: HistoryCard[] = [
  {
    id: "gauss",
    part: 1,
    year: "1827",
    photo: { file: "gauss.webp", author: "Christian Albrecht Jensen", licence: PD, source: commons("Carl Friedrich Gauss 1840 by Jensen.jpg") },
    sources: [
      { label: "Theorema Egregium (Wikipedia)", url: "https://en.wikipedia.org/wiki/Theorema_Egregium" },
      { label: "MacTutor: Carl Friedrich Gauss", url: "https://mathshistory.st-andrews.ac.uk/Biographies/Gauss/" },
    ],
  },
  {
    id: "riemann",
    part: 1,
    year: "1854",
    photo: { file: "riemann.webp", author: "Unknown photographer", licence: PD, source: commons("Georg Friedrich Bernhard Riemann.jpeg") },
    sources: [
      { label: "Riemann, On the Hypotheses… (Clifford's translation, TCD)", url: "https://www.maths.tcd.ie/pub/HistMath/People/Riemann/Geom/" },
      { label: "MacTutor: Bernhard Riemann", url: "https://mathshistory.st-andrews.ac.uk/Biographies/Riemann/" },
    ],
  },
  {
    id: "christoffel",
    part: 1,
    year: "1869",
    photo: {
      file: "christoffel.webp",
      author: "Unknown photographer (ETH-Bibliothek Zürich)",
      licence: PD,
      source: commons("ETH-BIB-Christoffel, Elwin Bruno (1829-1900)-Portrait-Portr 05361.tif"),
    },
    sources: [
      { label: "Christoffel 1869, J. reine angew. Math. 70 (DOI)", url: "https://doi.org/10.1515/crll.1869.70.46" },
      { label: "MacTutor: Elwin Bruno Christoffel", url: "https://mathshistory.st-andrews.ac.uk/Biographies/Christoffel/" },
    ],
  },
  {
    id: "ricci",
    part: 1,
    year: "1884–1894",
    photo: { file: "ricci.webp", author: "Unknown photographer", licence: PD, source: commons("Ricci-Curbastro.jpg") },
    sources: [
      { label: "MacTutor: Gregorio Ricci-Curbastro", url: "https://mathshistory.st-andrews.ac.uk/Biographies/Ricci-Curbastro/" },
      { label: "Ricci calculus (Wikipedia)", url: "https://en.wikipedia.org/wiki/Ricci_calculus" },
    ],
  },
  {
    id: "methodes",
    part: 1,
    year: "1900",
    photo: { file: "methodes.webp", author: "Unknown photographer", licence: PD, source: commons("Felix Christian Klein.jpg") },
    sources: [
      { label: "Ricci & Levi-Civita 1900, Math. Ann. 54, 125–201 (DOI)", url: "https://doi.org/10.1007/BF01454201" },
      { label: "MacTutor: Ricci-Curbastro (Klein's request)", url: "https://mathshistory.st-andrews.ac.uk/Biographies/Ricci-Curbastro/" },
    ],
  },
  {
    id: "voigt",
    part: 2,
    year: "1898",
    photo: { file: "voigt.webp", author: "Unknown photographer", licence: PD, source: commons("Woldemar Voigt, c. 1908.jpg") },
    sources: [
      { label: "Tensor: history (Wikipedia)", url: "https://en.wikipedia.org/wiki/Tensor" },
      { label: "Woldemar Voigt (Wikipedia)", url: "https://en.wikipedia.org/wiki/Woldemar_Voigt" },
      { label: "Cauchy stress tensor (Wikipedia)", url: "https://en.wikipedia.org/wiki/Cauchy_stress_tensor" },
    ],
  },
  {
    id: "minkowski",
    part: 2,
    year: "1908",
    photo: { file: "minkowski.webp", author: "Unknown photographer (frontispiece of Raum und Zeit, 1909)", licence: PD, source: commons("De Raum zeit Minkowski Bild.jpg") },
    sources: [
      { label: "Minkowski, Space and Time (Wikisource translation)", url: "https://en.wikisource.org/wiki/Translation:Space_and_Time" },
      { label: "Electromagnetic tensor (Wikipedia)", url: "https://en.wikipedia.org/wiki/Electromagnetic_tensor" },
    ],
  },
  {
    id: "grossmann",
    part: 2,
    year: "1912",
    photo: {
      file: "grossmann.webp",
      author: "Unknown photographer (ETH-Bibliothek Zürich)",
      licence: PD,
      source: commons("ETH-BIB-Grossmann, Marcel (1878-1936)-Portrait-Portr 01239.tif"),
    },
    sources: [
      { label: "T. Sauer, Marcel Grossmann and his contribution to GR (arXiv)", url: "https://arxiv.org/abs/1312.4068" },
      { label: "Marcel Grossmann (Wikipedia)", url: "https://en.wikipedia.org/wiki/Marcel_Grossmann" },
    ],
  },
  {
    id: "entwurf",
    part: 2,
    year: "1913",
    photo: { file: "entwurf.webp", author: "Jan F. Langhans", licence: PD, source: commons("Einstein Portr 05936.jpg") },
    sources: [
      { label: "History of general relativity (Wikipedia)", url: "https://en.wikipedia.org/wiki/History_of_general_relativity" },
      { label: "T. Sauer on the Einstein–Grossmann collaboration (arXiv)", url: "https://arxiv.org/abs/1312.4068" },
    ],
  },
  {
    id: "levicivita",
    part: 2,
    year: "1915–1917",
    photo: { file: "levicivita.webp", author: "Unknown photographer", licence: PD, source: commons("Levi-civita.jpg") },
    sources: [
      { label: "T. Damour, Levi-Civita and Einstein (slides, Padova 2018)", url: "https://events.math.unipd.it/Levi-Civita2018/sites/default/files/Damour_Slides.pdf" },
      { label: "MacTutor: Tullio Levi-Civita", url: "https://mathshistory.st-andrews.ac.uk/Biographies/Levi-Civita/" },
    ],
  },
  {
    id: "field",
    part: 2,
    year: "1915",
    photo: {
      file: "field.webp",
      author: "Ferdinand Schmutzer (restoration: Adam Cuerden)",
      licence: PD,
      source: commons("Einstein 1921 by F Schmutzer - restoration (cropped).jpg"),
    },
    sources: [
      { label: "Einstein 1915, Die Feldgleichungen der Gravitation (Wikisource)", url: "https://de.wikisource.org/wiki/Die_Feldgleichungen_der_Gravitation" },
      { label: "Einstein 1916, Annalen der Physik 49 (DOI)", url: "https://doi.org/10.1002/andp.19163540702" },
      { label: "Einstein field equations (Wikipedia)", url: "https://en.wikipedia.org/wiki/Einstein_field_equations" },
    ],
  },
  {
    id: "cuda",
    part: 3,
    year: "2006–2007",
    photo: { file: "cuda.webp", author: "Hyins", licence: PD, source: commons("NVIDIA GeForce 8800 GTX Board.jpg") },
    sources: [
      { label: "CUDA (Wikipedia)", url: "https://en.wikipedia.org/wiki/CUDA" },
      { label: "NVIDIA CUDA Toolkit archive (1.0, June 2007)", url: "https://developer.nvidia.com/cuda-toolkit-archive" },
      { label: "Nvidia: history (Wikipedia)", url: "https://en.wikipedia.org/wiki/Nvidia" },
    ],
  },
  {
    id: "imagenet",
    part: 3,
    year: "2007–2009",
    photo: {
      file: "imagenet.webp",
      author: "ITU Pictures",
      licence: "CC BY 2.0",
      licenceUrl: "https://creativecommons.org/licenses/by/2.0/",
      source: commons("Fei-Fei Li at AI for Good 2017.jpg"),
    },
    sources: [
      { label: "Deng et al., ImageNet, CVPR 2009 (DOI)", url: "https://doi.org/10.1109/CVPR.2009.5206848" },
      { label: "Russakovsky et al., ILSVRC, IJCV 2015 (arXiv)", url: "https://arxiv.org/abs/1409.0575" },
      { label: "ImageNet (Wikipedia)", url: "https://en.wikipedia.org/wiki/ImageNet" },
    ],
  },
  {
    id: "alexnet",
    part: 3,
    year: "2012",
    photo: {
      file: "alexnet.webp",
      author: "Fritzchens Fritz",
      ...CC0,
      source: commons("NVIDIA@40nm@Fermi@GF110@GeForce GTX 580@UA10B338 1041A1 N2Y540.000 GF110-375-A1 DSC04152 (27576804994).jpg"),
    },
    sources: [
      { label: "Krizhevsky, Sutskever & Hinton, NeurIPS 2012", url: "https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html" },
      { label: "Computer History Museum: AlexNet source code", url: "https://computerhistory.org/blog/chm-releases-alexnet-source-code/" },
    ],
  },
  {
    id: "cudnn",
    part: 3,
    year: "2014",
    drawing: "im2col",
    sources: [
      { label: "Chetlur et al., cuDNN (arXiv 2014)", url: "https://arxiv.org/abs/1410.0759" },
      { label: "NVIDIA cuDNN", url: "https://developer.nvidia.com/cudnn" },
    ],
  },
  {
    id: "tensorflow",
    part: 3,
    year: "2015",
    drawing: "dataflow",
    sources: [
      { label: "Abadi et al., TensorFlow white paper (arXiv)", url: "https://arxiv.org/abs/1603.04467" },
      { label: "tensorflow.org", url: "https://www.tensorflow.org/" },
    ],
  },
  {
    id: "tpu",
    part: 3,
    year: "2016",
    photo: {
      file: "tpu.webp",
      author: "Zinskauf",
      licence: "CC BY-SA 4.0",
      licenceUrl: "https://creativecommons.org/licenses/by-sa/4.0/",
      source: commons("Tensor Processing Unit 3.0.jpg"),
    },
    sources: [
      { label: "Jouppi et al., In-Datacenter Performance Analysis of a TPU (ISCA 2017)", url: "https://arxiv.org/abs/1704.04760" },
      {
        label: "Google Cloud: an in-depth look at the first TPU",
        url: "https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu",
      },
    ],
  },
  {
    id: "nvlink",
    part: 3,
    year: "2016–2018",
    photo: {
      file: "nvlink.webp",
      author: "Fritzchens Fritz",
      ...CC0,
      source: commons("Nvidia@16nm@Pascal@GP100@Tesla P100@T Taiwan 1912A1 PN9G70.S6W GP100-897-A1 DSC07141-DSC07141.jpg"),
    },
    sources: [
      { label: "NVIDIA Pascal (P100) white paper", url: "https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf" },
      { label: "NVIDIA NVSwitch technical overview", url: "https://images.nvidia.com/content/pdf/nvswitch-technical-overview.pdf" },
      { label: "NVIDIA NVLink", url: "https://www.nvidia.com/en-us/data-center/nvlink/" },
    ],
  },
  {
    id: "dgx1",
    part: 3,
    year: "2016",
    photo: {
      file: "dgx1.webp",
      author: "Peter Dasilva / European Union",
      licence: "CC BY 4.0",
      licenceUrl: "https://creativecommons.org/licenses/by/4.0/",
      source: commons("Jensen Huang (cropped) (2024).jpg"),
    },
    sources: [
      { label: "TOP500: NVIDIA delivers DGX-1 to OpenAI (2016)", url: "https://top500.org/news/nvidia-delivers-dgx-1-supercomputer-in-a-box-to-openai/" },
      { label: "Nvidia DGX (Wikipedia)", url: "https://en.wikipedia.org/wiki/Nvidia_DGX" },
    ],
  },
  {
    id: "pytorch",
    part: 3,
    year: "2016–2017",
    drawing: "tape",
    sources: [
      { label: "Paszke et al., PyTorch, NeurIPS 2019 (arXiv)", url: "https://arxiv.org/abs/1912.01703" },
      { label: "PyTorch Foundation", url: "https://pytorch.org/foundation" },
    ],
  },
  {
    id: "v100",
    part: 3,
    year: "2017",
    drawing: "tensorcore",
    sources: [
      { label: "NVIDIA Volta (V100) white paper", url: "https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf" },
      { label: "NVIDIA: Programming Tensor Cores in CUDA 9", url: "https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/" },
    ],
  },
  {
    id: "m1",
    part: 3,
    year: "2020",
    photo: { file: "m1.webp", author: "Henriok", ...CC0, source: commons("Apple M1.jpg") },
    sources: [
      { label: "Apple Newsroom: Apple unleashes M1 (2020)", url: "https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/" },
      { label: "corsix/amx: Apple AMX instruction set (reverse-engineered)", url: "https://github.com/corsix/amx" },
      { label: "PyTorch: MPS backend", url: "https://pytorch.org/docs/stable/notes/mps.html" },
    ],
  },
  {
    id: "einsum",
    part: 3,
    year: null,
    drawing: "einsum",
    sources: [
      { label: "torch.einsum", url: "https://pytorch.org/docs/stable/generated/torch.einsum.html" },
      { label: "numpy.einsum", url: "https://numpy.org/doc/stable/reference/generated/numpy.einsum.html" },
      { label: "Einstein notation (Wikipedia)", url: "https://en.wikipedia.org/wiki/Einstein_notation" },
    ],
  },
];

export const HISTORY_PARTS: Part[] = [1, 2, 3];
