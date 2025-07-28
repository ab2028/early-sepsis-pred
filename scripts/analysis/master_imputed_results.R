# ===================================================================
# Master Script: Combine Results from All Models (5-Fold CV)
# ===================================================================

# ---- Step 0: Load Required Packages ----
library(data.table)

# ---- Step 1: Source Individual Model Scripts ----
suppressMessages({
  suppressWarnings({
    # ---- Tree-based models ----
    source("rf_imputed_5cv.R", echo = FALSE)         # produces final_test_results
    rf_results <- copy(final_test_results)
    rf_results[, Model := "Random Forest"]
    
    source("xgboost_imputed_5cv.R", echo = FALSE)    # produces final_test_results
    xgb_results <- copy(final_test_results)
    xgb_results[, Model := "XGBoost"]
    
    source("lightgbm_imputed_5cv.R", echo = FALSE)   # produces final_test_results
    lgb_results <- copy(final_test_results)
    lgb_results[, Model := "LightGBM"]
    
    # ---- Classical models ----
    source("knn_5cv.R", echo = FALSE)                      # produces final_test_results
    knn_results <- copy(final_test_results)
    knn_results[, Model := "KNN"]
    
    source("svm_5cv.R", echo = FALSE)                      # produces final_test_results
    svm_results <- copy(final_radial_test_results)
    svm_results[, Model := "SVM (Radial)"]
    
    source("naivebayes_5cv.R", echo = FALSE)               # produces final_test_results
    nb_results <- copy(final_test_results)
    nb_results[, Model := "Naive Bayes"]
    
    source("LRlogit_5cv.R", echo = FALSE) # produces final_test_results
    logit_results <- copy(final_test_results)
    logit_results[, Model := "LR (Logit)"]
  })
})


# ---- Step 2: Combine Results ----
all_results <- rbindlist(list(
  rf_results, xgb_results, lgb_results,
  knn_results, svm_results, nb_results, logit_results
), fill = TRUE)

# ---- Step 3: Summarize Average Metrics Per Model ----
metric_cols <- c("Accuracy", "AUC", "AUPRC", "Sensitivity", 
                 "Specificity", "Precision", "F1_Score")

combined_summary <- all_results[, lapply(.SD, mean, na.rm = TRUE), 
                                by = Model, .SDcols = metric_cols]

# ---- Step 4: Round Metrics (for clean output or saving) ----
for (col in metric_cols) {
  combined_summary[[col]] <- as.numeric(combined_summary[[col]])
  combined_summary[[col]] <- round(combined_summary[[col]], 4)
}

cat("\n\n================= FINAL COMBINED SUMMARY (ALL MODELS) =================\n")
print(
  combined_summary[, .(Model, 
                       Accuracy = round(Accuracy, 4),
                       AUC = round(AUC, 4),
                       AUPRC = round(AUPRC, 4),
                       Sensitivity = round(Sensitivity, 4),
                       Specificity = round(Specificity, 4),
                       Precision = round(Precision, 4),
                       F1_Score = round(F1_Score, 4))]
)

# ---- Step 5: (Optional) Save Combined Results to CSV ----
# fwrite(combined_summary, "combined_model_results_all.csv")

# ===================================================================
# End of Script
# ===================================================================
