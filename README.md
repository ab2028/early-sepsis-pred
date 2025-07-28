# early-sepsis-pred
This repository contains the code and analysis for the project:
"PAPER TITLE", a benchmarking study of classical machine learning models for early sepsis prediction using ICU electronic health record (EHR) data.

Sepsis is a leading cause of ICU mortality, and early prediction can dramatically improve patient outcomes. In this project, we systematically compared seven supervised machine learning algorithms (Random Forest, XGBoost, LightGBM, kNN, Naive Bayes, SVM, Logistic Regression) across four distinct feature engineering strategies, highlighting best practices for early sepsis detection.

## Dataset
Due to size and licensing constraints, the raw datasets are not stored in this repository. We use the PhysioNet/Computing in Cardiology Challenge 2019 dataset, which can be downloaded from PhysioNet: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
Once downloaded, place the raw .csv files into the `data/raw` folder.

Run the preprocessing scripts (below) to generate cleaned and engineered datasets.

## 2. Setup

### Clone the repository
```
git clone https://github.com/<your-username>/early-sepsis-pred.git
cd early-sepsis-pred
```

### Install dependencies
Ensure you have R (≥4.0) installed, along with the following packages:
```r
install.packages(c("data.table", "dplyr", "tidyr", "pROC", "PRROC", "caret", "ranger",
                   "caTools", "xgboost", "lightgbm", "e1071", "SHAPforxgboost"))
```

 ## 3. Run Preprocessing
 Run the scripts in `scripts/preprocessing/` to clean and engineer features. Example:
 ```r
setwd("path/to/early-sepsis-pred")
source("scripts/preprocessing/clean_raw_data.R")
source("scripts/preprocessing/create_24h_summary_dataset.R")
```
The file `create_baseline_dataset.R` includes a commented out line 

## 4. Train Models
Each model script will train using 5-fold cross-validation and output metrics to the console.
Example:
```r
source(scripts/models/xgboost_imputed_5cv.R)
```
You can also run the master scripts to train all models and see combined results in a table:
```r
source("results/master_imputed_results.R")
source("results/master_nonimputed_results.R")
```

## 5. View Results and Plots
Metrics are printed to the R console by default. Plots (e.g., SHAP plots, histograms) will open in the R plotting window. 

## 6. Project Structure
```graphql
early-sepsis-pred/
├── data/
│   ├── raw/            # Place raw PhysioNet CSVs here
│   └── processed/      # Preprocessed datasets (generated locally)
├── scripts/
│   ├── preprocessing/  # Data cleaning & feature engineering scripts
│   ├── models/         # Individual ML model training scripts
│   └── results/        # Master scripts for combined metrics
├── results/            # (Optional) Saved metrics and plots
└── README.md
```
