# MLiB: Interpretable multi-omic machine learning

Framework for multi-omic model-based data integration for the retention of interpretability via recursive feature elimination. This repository was made for the individual project of the course "CS4260: Machine Learning in Bioinformatics" at TU Delft. Below follows and explanation of the repo structure. The input folder in `data` has been omitted due to file size limitations on GitHub.

- `src`: All code used for the project. All functions in each script are numbered.
  - `_1_process_pancan_data.py`: Used to pre-process the PANCAN data.
  - `_2_scaling_and_dim_reduction.py`: Used for scaling and dimensionality reduction. This file was also used to tune the models on the full-feature datasets.
  - `_3_voting_and_importances.py`: This file was used to implement the voting strategies and compute the permutation importances of the features in the 500 features datasets. This script was also used to tune the models and evaluate the test performance.
- `data`: This folder contains all data used in the project
  - `output`:
    - `rfe`: Training performances and most important features for each dataset
      - `tuning`: hyperparameter tuning results on the full-feature datasets.
    - `tuning`: Hyperparamter tuning results on the 500 feature datasets.
    - `permutation_importance`: Permutation importances features in the 500 feature datsets.
    - `models`: Models trained on the 500-feature training datasets.
- `images`: Images used in the report
