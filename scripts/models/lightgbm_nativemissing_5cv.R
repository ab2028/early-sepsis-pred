# ===================================================================
# Sepsis Prediction with LightGBM, 5-Fold CV, and Undersampling
# - Native handling of missing values in predictors (no imputation)
# - Manual undersampling (50/50) within each fold
# ===================================================================

# ---- Step 1: Load Required Packages ----
library(data.table)   # Efficient data manipulation
library(lightgbm)     # LightGBM implementation
library(dplyr)        # Data wrangling verbs
library(pROC)         # AUC calculation
library(PRROC)        # AUPRC calculation
library(caret)        # For createFolds and confusionMatrix

# ---- Step 2: Load and Prepare Data ----
# source("create_6hr_summary_dataset.R") #change according to which dataset is used

sepsis.data <- data.table(final_df)
sepsis.data <- sepsis.data[!is.na(SepsisLabel)]   # Remove rows with missing labels

if ("Patient_ID" %in% names(sepsis.data)) {
  sepsis.data[, Patient_ID := NULL]   # Remove Patient_ID column if present
}
sepsis.data[, SepsisLabel := as.numeric(as.character(SepsisLabel))]

# ---- Step 3: Create Performance Calculation Function ----
calculate_performance_metrics <- function(true_labels, pred_probs, threshold = 0.5) {
  valid_indices <- !is.na(true_labels) & !is.na(pred_probs)
  clean_labels <- true_labels[valid_indices]
  clean_probs <- pred_probs[valid_indices]
  
  if (length(unique(clean_labels)) < 2) {
    return(data.table(Accuracy=NA, Sensitivity=NA, Specificity=NA, 
                      Precision=NA, F1_Score=NA, AUC=NA, AUPRC=NA))
  }
  
  auc_score <- as.numeric(pROC::auc(clean_labels, clean_probs, quiet = TRUE))
  pr_obj <- pr.curve(scores.class0 = clean_probs[clean_labels == 1], 
                     scores.class1 = clean_probs[clean_labels == 0], curve = FALSE)
  auprc_score <- pr_obj$auc.integral
  
  pred_labels <- ifelse(clean_probs > threshold, 1, 0)
  cm_result <- confusionMatrix(factor(pred_labels, levels = c(0, 1)),
                               factor(clean_labels, levels = c(0, 1)),
                               positive = "1")
  
  data.table(
    Accuracy = cm_result$overall['Accuracy'],
    AUC = auc_score,
    AUPRC = auprc_score,
    Sensitivity = cm_result$byClass['Sensitivity'],
    Specificity = cm_result$byClass['Specificity'],
    Precision = cm_result$byClass['Pos Pred Value'],
    F1_Score = cm_result$byClass['F1']
  )
}

# ---- Step 4: Set up 5-Fold Cross-Validation ----
set.seed(123)
folds <- createFolds(sepsis.data$SepsisLabel, k = 5, list = TRUE, returnTrain = FALSE)
test_performance_list <- list()

# ---- Step 5: Loop Through Folds (Undersampling, Training, Evaluation) ----
for (i in 1:length(folds)) {
  cat(sprintf("\n\n================= Processing Fold %d/%d =================\n", i, length(folds)))
  
  test_indices <- folds[[i]]
  train_df <- sepsis.data[-test_indices, ]
  test_df <- sepsis.data[test_indices, ]
  
  # --- Step 5.1: Undersample negatives (50/50 balance) ---
  sepsis_cases <- train_df[SepsisLabel == 1]
  no_sepsis_cases <- train_df[SepsisLabel == 0]
  
  if (nrow(sepsis_cases) == 0) {
    cat("No positive cases in this training fold. Skipping...\n")
    next
  }
  
  set.seed(42)
  sampled_negatives <- no_sepsis_cases[sample(.N, size = nrow(sepsis_cases))]
  train_balanced <- rbindlist(list(sepsis_cases, sampled_negatives))
  
  cat(sprintf("Training data balanced: %d Sepsis, %d No Sepsis\n", 
              sum(train_balanced$SepsisLabel == 1), 
              sum(train_balanced$SepsisLabel == 0)))
  
  # --- Step 5.2: Prepare LightGBM Data Matrices ---
  predictor_cols <- setdiff(names(train_balanced), "SepsisLabel")
  x_train <- as.matrix(train_balanced[, ..predictor_cols])
  y_train <- train_balanced$SepsisLabel
  x_test  <- as.matrix(test_df[, ..predictor_cols])
  y_test  <- test_df$SepsisLabel
  
  dtrain <- lgb.Dataset(data = x_train, label = y_train)
  
  # --- Step 5.3: Train the LightGBM Model ---
  params <- list(
    objective = "binary", metric = "auc", boosting = "gbdt", 
    num_leaves = 11, max_depth = 5, lambda_l2 = 0.5, lambda_l1 = 0.1,
    min_child_samples = 50, learning_rate = 0.01, verbosity = -1
  )
  
  lgbm_model <- lgb.train(
    params = params, data = dtrain, nrounds = 500, verbose = -1
  )
  
  # --- Step 5.4: Evaluate on TEST data only ---
  test_preds <- predict(lgbm_model, x_test)
  test_results <- calculate_performance_metrics(y_test, test_preds)
  test_results[, Fold := i]
  test_performance_list[[i]] <- test_results
  
  cat(sprintf("Fold %d - Test AUC: %.4f\n", i, test_results$AUC))
}

# ---- Step 6: Summarize TEST Results ----
final_test_results <- rbindlist(test_performance_list, fill = TRUE)
metric_cols <- c("Accuracy", "AUC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1_Score")

cat("\n\n================= FINAL LightGBM 5-FOLD CV SUMMARY =================\n")
if (nrow(final_test_results) > 0) {
  cat("\n--- Average TEST Performance Across 5 Folds ---\n")
  avg_test_metrics <- final_test_results[, lapply(.SD, mean, na.rm = TRUE), .SDcols = metric_cols]
  print(round(avg_test_metrics, 4))
  
  cat("\n--- Detailed Test Results Per Fold ---\n")
  print(final_test_results)
} else {
  cat("No folds completed successfully.\n")
}

# ===================================================================
# End of Script
# ===================================================================
