# early-sepsis-pred
This repository contains the code and analysis for the project:
"PAPER TITLE", a benchmarking study of classical machine learning models for early sepsis prediction using ICU electronic health record (EHR) data.

Sepsis is a leading cause of ICU mortality, and early prediction can dramatically improve patient outcomes. In this project, we systematically compared seven supervised machine learning algorithms (Random Forest, XGBoost, LightGBM, kNN, Naive Bayes, SVM, Logistic Regression) across four distinct feature engineering strategies, highlighting best practices for early sepsis detection.

## Dataset
We use the PhysioNet/Computing in Cardiology Challenge 2019 dataset, which can be downloaded from PhysioNet: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis

👉 Download dataset from PhysioNet

Important:

Raw data is not included in this repository due to size and licensing.

Once downloaded, place the raw .csv files into the data/raw/ folder.

Run the preprocessing scripts (below) to generate cleaned and engineered datasets.
