# Data sources

Everything here was fetched on 2026-11-05 into a separate folder outside the repository, treated as untrusted data
(read with numpy, PIL and csv only, Python run with `-I`, nothing from the downloads executed), then reduced to the
small files below by `data/make_subset.py`. No network access happens at run time.

## nih_cxr_subset.npz (NIH ChestX-ray14, NIH Clinical Center)

- **What**: 549 frontal (PA) chest radiographs, one per patient, downsampled to 64×64 grayscale uint8, with six
  findings kept as a multi-hot vector: cardiomegaly, effusion, infiltration, atelectasis, pneumothorax, and mass or
  nodule (merged), plus "No Finding". Counts: no finding 150, cardiomegaly 87, effusion 85, infiltration 103,
  atelectasis 86, pneumothorax 37, mass/nodule 86 (an image can carry two). 410 training, 139 test images
  (stratified by the first finding). Arrays: `images` (N,64,64) uint8, `labels` (N,) int8 (0 = no finding, 1 + index
  of the first finding, for the data viewer), `findings` (N,6) uint8, `split` (N,) uint8 (0 train, 1 test),
  `image_index` (N,) NIH file name, `patient_id` (N,) int32. About 2.0 MB.
- **Source**: NIH Clinical Center, ChestX-ray14 (ChestX-ray8 extended), https://nihcc.app.box.com/v/ChestXray-NIHCC.
  Labels from `Data_Entry_2017_v2020.csv` (SHA-256 `c69a6daca3549af707ca9cacbdf9f9a7b6a9188e8c61157df653520bb72d8eb1`).
  The images were meant to be fetched one by one from the public Google Cloud copy
  (`gcs-public-data--healthcare-nih-chest-xray`), but that bucket refuses anonymous reads (HTTP 401/403), so the
  same files were taken from NIH's own download site: the first two image archives (`images_001.tar.gz`,
  `images_002.tar.gz`, the NIH Box static links of NIH's `batch_download_zips.py`) were streamed and only the
  selected PNGs written to disk; nothing else in the archives was kept.
- **Selection**: from the CSV, for each patient the first PA image whose findings all lie among the six (or "No
  Finding"); then, in archive order, an image is kept while one of its labels is below its quota (150 for "No
  Finding", 90 per finding); finally images with more than two of the six findings are dropped. Pneumothorax stays
  rare (37) because few patients' first eligible PA image carries it.
- **Terms** (verbatim, "License and attribution", Google Cloud's page for this public NIH dataset,
  https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest, checked 2026-11-06):
  "There are no restrictions on the use of the NIH chest x-ray images. However, the dataset has the following
  attribution requirements: Provide a link to the NIH download site: https://nihcc.app.box.com/v/ChestXray-NIHCC;
  Include a citation to the CVPR 2017 paper; Acknowledge that the NIH Clinical Center is the data provider."
  **We thank the NIH Clinical Center for providing this dataset.**
- **Citation**: X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri and R. M. Summers, "ChestX-ray8: Hospital-scale Chest X-ray
  Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases", IEEE CVPR
  2017, pp. 2097–2106 (arXiv:1705.02315).
- **Caveat**: the labels were mined from radiology reports by natural-language processing and are noisy (the
  authors estimate above 90 % accuracy overall, i.e. roughly 10 % or more wrong, worse for some findings); at 64×64
  pixels much of the evidence is gone. The numbers of this toy say nothing about clinical accuracy.

## pmc_cxr_subset.npz, pmc_captions.csv, ATTRIBUTION.csv (PubMed Central, CC BY)

- **What**: 316 chest radiograph figures from 316 open-access articles in PubMed Central, each downsampled to 64×64
  grayscale uint8 (`images`), with `labels` (N,) int8 (0 = no finding keyword in the caption, else 1 + the first
  keyword finding, for the data viewer only) and `split` (N,) uint8 (251 train, 65 test, `default_rng(1)`).
  `pmc_captions.csv` (id, pmcid, figure_id, split, caption, report_text, text_source) holds each figure's caption as
  written in the article and the stage-2 text `report_text`: the caption followed by the same article's own body
  sentences that cite that figure (an `<xref ref-type="fig">` to it), then, if room is left, body sentences that
  mention a chest radiograph; whole sentences only, at most 120 words. These sentences were extracted from the
  article XML re-fetched from PMC (E-utilities, the same accepted CC BY articles; 315 of 316 articles have at least
  one sentence citing the figure, 179 also contribute sentences mentioning a chest radiograph, 1 has the caption
  only). The code lower-cases the text, removes figure labels, "(Fig. 2)"-style references and panel letters, and
  keeps whole sentences up to 360 characters (the context of the tiny LM); no text needed a cut inside its first
  sentence. Words per text: caption alone median 12 (IQR 8–18); raw caption + sentences median
  48 (IQR 31–80); the stage-2 target after the 360-character limit median 35 (IQR 26–45). (The first version of this
  problem used only the first clause of the caption, ≤ 50 characters: median 6 words.)
- **Licences**: every figure comes from an article whose own JATS XML licence (`<permissions><license>`, fetched
  from PMC through NCBI E-utilities `efetch db=pmc`) is CC BY (4.0: 140, 3.0: 132, 2.0: 42, 2.5: 2). The rule was:
  keep CC BY, CC BY-SA or CC0 of any version; reject NC, ND, any other licence and a missing one. **Every figure and
  caption is credited in `ATTRIBUTION.csv`** (PMCID, figure id and label, authors, article title, journal, year,
  licence, article and image links, fetch date, and the change note: image downsampled to 64×64, cropped, converted
  to grayscale; text = the figure caption plus the article's own sentences citing the figure). The captions and
  sentences are reproduced unchanged (in `pmc_captions.csv`) under the same licence.
- **How they were found**: the metadata CSVs of ROCOv2 (Rückert et al. 2024, Zenodo record 10821435;
  `license_information.csv`, `cui_mapping.csv`, `{train,valid,test}_concepts_manual.csv`) were used ONLY as an
  index: image ids whose manual UMLS concepts include both "Plain x-ray" (C1306645) and "Chest" (C0817096) gave 5663
  candidate PMCIDs. **Nothing from ROCOv2 is bundled or used otherwise** (the ROCOv2 compilation is CC BY-NC 4.0):
  not its images, captions, concepts or attribution strings; the licence, captions and figures all come from PMC.
- **Licence checks** (random order of the candidates, seed 0, stopped at 430 downloaded figures): 2114 articles
  checked; 1575 accepted (CC BY 4.0: 671, 3.0: 607, 2.0: 280, 2.5: 16; CC0: 1), 539 rejected by licence (CC BY-NC
  4.0: 284, 3.0: 249, 2.0: 6), 0 missing or other; 1 PMCID returned no XML. Of the accepted, 648 had no figure
  caption naming a chest radiograph without another modality (CT, MRI, ultrasound, lateral view, …), and 497 figure
  pages answered with an automated-access check page instead of the article, which was not worked around, so those
  were skipped. Of the 430 figures downloaded, 112 failed the visual sanity filter (colour 76, aspect ratio 31,
  mostly white or black 5) and 2 were removed by eye (a four-panel segmentation figure, a skull radiograph whose
  caption said "PA view"): 316 kept. Requests to NCBI were limited to 3 per second with a generic tool name.
- **Caveat**: captions are short, heterogeneous and written for readers of a case report ("chest x-ray showing
  ..."), often refer to arrows or panels, and describe a single selected image; they are not radiology reports.

## Why no real radiology reports

None of the openly available report collections passes the licence rule used here (open, redistributable,
commercial use allowed): IU X-ray / OpenI (Demner-Fushman et al. 2016) is CC BY-NC-ND; MIMIC-CXR (Johnson et al.
2019), VinDr-CXR and BRAX require credentialed access under data use agreements; CheXpert and PadChest are licensed
for research use only. So stage 1 uses template reports generated in code (`tensors.report`) from the NIH label
sets, and stage 2 uses CC BY figure captions.

## Template reports (generated, no data file)

Each label set with at most two of the six findings (22 sets) is written in up to 8 phrasings: two per finding
("cardiomegaly." / "enlarged heart.", "pleural effusion." / "effusion.", "infiltrate." / "consolidation.",
"atelectasis." / "volume loss.", "pneumothorax." / "small pneumothorax.", "nodule." / "mass."), "normal chest." or
"clear lungs." when there is none, and optionally "no pneumothorax." at the end. `data/template_reports.txt` named in
training.smk is a label: the reports are generated in memory.

## training.smk

The training process as a Snakefile-style rule graph (written for this problem). It is parsed by the tensor DAG
engine and never run by Snakemake; its input/output names are labels, no such files are written.

## Checksums (SHA-256 of the bundled files; see `make_subset.py` to rebuild them)

The checksums of the bundled files are listed in `SHA256SUMS.txt` next to this file.
