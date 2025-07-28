# early-sepsis-pred
This repository contains the code and analysis for the project:
"PAPER TITLE", a benchmarking study of classical machine learning models for early sepsis prediction using ICU electronic health record (EHR) data.

Sepsis is a leading cause of ICU mortality, and early prediction can dramatically improve patient outcomes. In this project, we systematically compared seven supervised machine learning algorithms (Random Forest, XGBoost, LightGBM, kNN, Naive Bayes, SVM, Logistic Regression) across four distinct feature engineering strategies, highlighting best practices for early sepsis detection.

## Dataset
Due to size and licensing constraints, the raw datasets are not stored in this repository. We use the PhysioNet/Computing in Cardiology Challenge 2019 dataset, which can be downloaded from PhysioNet: https://www.kaggle.com/datasets/salikhussaini49/prediction-of-sepsis
Once downloaded, place the raw .csv files into the `data/raw` folder.

Run the preprocessing scripts (below) to generate cleaned and engineered datasets.

##Setup

###Clone the repository
`git clone https://github.com/<your-username>/early-sepsis-pred.git
cd early-sepsis-pred`
