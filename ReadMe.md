# The intrinsic geometry of reading

This repository contains the code and notebooks required to run the analysis and generate the figures presented in our manuscript "The intrinsic geometry of reading."

## Dependencies

### Python packages

Install all Python dependencies via:

```bash
pip install -r requirements.txt
```

Two dependencies are not on PyPI and are installed directly from GitHub. They are included in `requirements.txt` as editable installs, but note them explicitly here:

- **[brain2behaviour](https://github.com/neurabenn/brain2behaviour)** — core dataset class and CPM utilities used throughout the analysis
- **[surfdist](https://github.com/neurabenn/surfdist)** — cortical surface geodesic distance computation

If `pip install -r requirements.txt` does not install them (e.g. in a fresh environment), install manually:

```bash
pip install git+https://github.com/neurabenn/brain2behaviour.git
pip install git+https://github.com/neurabenn/surfdist.git
```

### HCP data

Several files required to recreate the analysis are not ours to distribute, but are freely available through the Human Connectome Project:

1. **HCP unrestricted behavioral data** — download [here](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release)
2. **HCP restricted behavioral data** — access instructions [here](https://www.humanconnectome.org/study/hcp-young-adult/document/restricted-data-usage)
   - Required to respect family structure in cross-validation fold splits
   - Also contains confounds: age, height, weight, blood pressure

## Data

The analysis data (parcellated surface area, cortical distance matrices, and functional connectivity matrices) are hosted on Zenodo due to file size:

**[https://zenodo.org/records/20558695](https://zenodo.org/records/20558695)**

Download and unpack the archive, then update the path variables at the top of each notebook to point to your local copy. In most notebooks this is the `data_dir` variable:

```python
data_dir = '/path/to/unpacked/data'
```

## Notebooks

### nb1 — Build the dataset
Sets up the analysis dataset using the `brain2behaviour` dataset class. Loads HCP behavioral and demographic data, selects confounds, and serializes subject-level data objects. The `data_dir` variable must point to your unpacked Zenodo data.

## Scripts to run models
These scripts use the datasets we build in nb1 to run the full CPM pipeline. 
Steps run (feature selection, permutation testing), parallelized cluster scripts are provided alongside the notebooks:
- `End2EndCPM_wPerms.py` — end-to-end CPM with permutation testing is the main script
   - Call it with a dataset the permutations and the task to be tested. 
- `select_features_batch.py` — batch feature selection across CV folds -- called by end2end
- `CollectFeaturesandPredict.py` — collects fold results and runs prediction -- called by end2end

Permutation exchangeability blocks for HCP data are precomputed and stored in `fold_permutations/`. For background on HCP-compatible permutation testing see [Winkler et al. 2015](https://brainder.org/2015/12/07/permutation-tests-in-the-human-connectome-project/) and the [PALM exchangeability block guide](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/PALM(2f)ExchangeabilityBlocks.html#EBs_for_data_of_the_Human_Connectome_Project).


### nb2 — Figure 1a, b
Runs significance calculations for model performance (cross-validation) and tests for differences between models. Plots Figure 1a and 1b. Based on outputs of CPM run with the dataset construction in nb1.

### nb3 — Figure 1c
Generates the circle plot and heatmap visualizations of stable selected features (Figure 1c). Features visualized are those with consistent associations across all folds of cross-validation.

### nb4 — Figure 1c surface visualization (gifti generation)
Projects CPM features onto the cortical surface and writes gifti metric files. To use the outputs: set the cortex structure and palette via Connectome Workbench (`wb_command -set-structure`, `wb_command -metric-palette`), or open `surfaces_and_scene_files/FeaturesSchaefer400.scene` directly in Connectome Workbench to view precalculated surface visualizations. Alternatively, load the workbench scene stored in the `surfaces_and_scene_files/` directory.

### nb5 — Figure 2
Shows how generalization from the discovery (HCP Young Adult) to the validation cohort (HCP Lifespan Aging) was done.

### nb6 — Supplementary Figures S2–S5
Replicates the supplementary figures. Due to file size limits on Zenodo, the model outputs used to calculate between-model comparisons and model significance are available upon request.

### nb7 — Supplementary Figure S6
Uses the dataset files to plot surface area, cortical distance, and functional connectivity stable features in the discovery cohort with the oral reading recognition test scores and confounds.