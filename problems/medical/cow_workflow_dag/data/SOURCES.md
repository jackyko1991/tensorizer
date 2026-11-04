# Data sources

Everything here is either derived from CC0 data or computed by us (CC0). Snakemake was run once, offline, in a
separate WSL2 environment outside the project; it is never run by the problem or its tests.

## cases/*.npz (six Circle of Willis crops)

Files: cases/sub-000.npz, cases/sub-002.npz, cases/sub-006.npz, cases/sub-021.npz, cases/sub-033.npz,
cases/sub-354.npz.

- **Source:** TopCoW 2024 data release, **MRA_Lausanne subset** (`MRA_Lausanne.zip`, md5-verified), Zenodo record
  doi:10.5281/zenodo.15692630. Files used per case: `imagesTr/<case>_ses-…_angio_LPS.nii.gz` (TOF-MRA),
  `cow_seg_labelsTr/<case>_…_LPS_Mask.nii.gz` (Circle of Willis multiclass labels) and `roi_loc_labelsTr/<case>_….txt`
  (the CoW region of interest).
- **Licence:** CC0 1.0 for this subset (as stated on the Zenodo record). The main TopCoW release (which restricts
  commercial use) and the IXI and ISLES subsets are not used.
- **Cite:** Yang K. et al., "The TopCoW Challenge: Topology-Aware Circle of Willis Segmentation for CT and MR
  Angiography", arXiv:2312.17670 (2023); and the Lausanne cohort:
  Di Noto T. et al., "Towards Automated Brain Aneurysm Detection in TOF-MRA: Open Data, Weak Labels, and Anatomical
  Knowledge", Neuroinformatics (2023, published online 2022), doi:10.1007/s12021-022-09597-0.
- **Cases:** sub-000, sub-002, sub-006, sub-021, sub-033, sub-354 (healthy controls), chosen for different anatomy:
  complete circles, a missing Acom label (sub-006, sub-021, sub-033), missing Pcom labels (sub-021), a third A2
  (sub-000).
- **Processing (ours, numpy only):** NIfTI read with its scaling (scl_slope, scl_inter); a box of 48 × 40 × 24 mm
  centred on the ROI centre, resampled to 0.5 mm isotropic (trilinear for the image, nearest neighbour for the labels)
  in the voxel axes of the LPS-oriented files (i → patient left, j → posterior, k → superior); intensity normalised
  per case as n = (I − p50)/(p99 − p50) over the crop, clipped to [0, 1.5], stored as uint8 = round(170 n); labels
  stored as uint8 (TopCoW classes 1 BA, 2 R-PCA, 3 L-PCA, 4 R-ICA, 5 R-MCA, 6 L-ICA, 7 L-MCA, 8 R-Pcom, 9 L-Pcom,
  10 Acom, 11 R-ACA, 12 L-ACA, 15 third A2). No scanner coordinates or anything identifying is kept.
- **Caveat:** labels that reach outside the crop are cut by it (about 65 to 90 % of each case's labelled volume is
  inside); thresholding TOF-MRA misses slow flow and thin vessels.

## Snakefile

The workflow, written by us (CC0). Each rule calls `python -m problems.medical.cow_workflow_dag.workflow`.

## snakemake/ (test references)

Made by us (CC0) with Snakemake 9.27.1 (snakemake-minimal, conda-forge/bioconda, Python 3.14, torch 2.14.1 CPU) in
WSL2 Ubuntu 24.04 on an AMD Ryzen 5 5600G (12 logical CPUs), one thread per job (OMP_NUM_THREADS=1):

- `dag.dot`: `snakemake --dag`
- `dryrun.txt`, `dryrun_jobs.csv`: `snakemake -n` and its job list in order
- `dryrun_after_touch_only.txt`: `snakemake -n` after `touch cases/sub-002.npz` (nothing to do: Snakemake compares
  the checksum of small input files, so a touch alone is not a change)
- `dryrun_after_touch.txt`, `rerun_after_touch.csv`: after rewriting cases/sub-002.npz with the same arrays (new bytes)
- `summary.tsv`: `snakemake --summary` after a complete run
- `timings_cores1.csv`, `timings_cores6.csv`, `timings_cores12.csv`: per-job start and end, stamped by each job's shell; `runs.csv`: whole-run wall clock
- `startup.csv`: the start-up of one Python process with torch (and with the job module), five repeats
- `cohort.csv`: the cohort table the Snakemake jobs wrote
