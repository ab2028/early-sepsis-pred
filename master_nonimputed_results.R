# ===================================================================
# Master Script: Combine Results from Multiple Models (5-Fold CV)
# ===================================================================

# ---- Step 0: Load Required Packages ----
library(data.table)

# ---- Step 1: Source Individual Model Scripts ----
suppressMessages({
  suppressWarnings({
    source("rf_nativemissing_5cv.R", echo = FALSE)       # produces final_test_results
    rf_results <- copy(final_test_results)
    rf_results[, Model := "Random Forest"]
    
    source("xgboost_nativemissing_5cv.R", echo = FALSE)  # produces final_test_results
    xgb_results <- copy(final_test_results)
    xgb_results[, Model := "XGBoost"]
    
    source("lightgbm_nativemissing_5cv.R", echo = FALSE) # produces test_results_df
    lgb_results <- copy(final_test_results)
    lgb_results[, Model := "LightGBM"]
  })
})

# ---- Step 2: Combine Results ----
all_results <- rbindlist(list(rf_results, xgb_results, lgb_results), fill = TRUE)

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

cat("\n\n================= FINAL COMBINED SUMMARY =================\n")
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
# fwrite(combined_summary, "combined_model_results.csv")


# ===================================================================
# End of Script
# ===================================================================
