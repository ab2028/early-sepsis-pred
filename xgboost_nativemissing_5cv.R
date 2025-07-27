# ===================================================================
# Sepsis Prediction with XGBoost, 5-Fold CV, and Undersampling
# - Native handling of missing values in predictors (no imputation)
# - Manual undersampling (50/50) within each fold
# ===================================================================

# ---- Step 0: Load Required Packages ----
library(data.table)
library(xgboost)
library(dplyr)
library(pROC)
library(PRROC)
library(caret) # For createFolds and confusionMatrix

# ---- Step 1: Load and Prepare Data ----
# Assumes 'final_df' is loaded from your feature engineering script.
sepsis.data <- data.table(final_df)
if ("Patient_ID" %in% names(sepsis.data)) {
  sepsis.data[, Patient_ID := NULL]
}
sepsis.data[, SepsisLabel := as.integer(as.character(SepsisLabel))]
sepsis.data <- sepsis.data[!is.na(SepsisLabel)]   # Remove rows with missing labels

# ---- Step 2: Create a Reusable Performance Calculation Function ----
calculate_performance_metrics <- function(true_labels, pred_probs, threshold = 0.5) {
  # Threshold-independent metrics
  auc_score <- as.numeric(auc(true_labels, pred_probs, quiet = TRUE))
  pr_obj <- pr.curve(
    scores.class0 = pred_probs[true_labels == 1], 
    scores.class1 = pred_probs[true_labels == 0], 
    curve = FALSE
  )
  auprc_score <- pr_obj$auc.integral
  
  # Threshold-dependent metrics
  pred_labels <- ifelse(pred_probs > threshold, 1, 0)
  cm_result <- try(confusionMatrix(
    data = factor(pred_labels, levels = c(0, 1)), 
    reference = factor(true_labels, levels = c(0, 1)), 
    positive = "1"), 
    silent = TRUE
  )
  
  if (inherits(cm_result, "try-error")) {
    accuracy <- NA; sensitivity <- NA; specificity <- NA; precision <- NA; f1 <- NA
  } else {
    accuracy <- cm_result$overall['Accuracy']
    sensitivity <- cm_result$byClass['Sensitivity']
    specificity <- cm_result$byClass['Specificity']
    precision <- cm_result$byClass['Pos Pred Value']
    f1 <- cm_result$byClass['F1']
  }
  
  return(data.table(
    Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity,
    Precision = precision, F1_Score = f1, AUC = auc_score, AUPRC = auprc_score
  ))
}

# ---- Step 3: Set up 5-Fold Cross-Validation ----
set.seed(123)
folds <- createFolds(sepsis.data$SepsisLabel, k = 5, list = TRUE, returnTrain = FALSE)
test_performance_list <- list()

# ---- Step 4: Loop Through Folds for Training, Undersampling, and Evaluation ----
for (i in 1:length(folds)) {
  cat(sprintf("\n\n================= Processing Fold %d/%d =================\n", i, length(folds)))
  
  test_indices <- folds[[i]]
  train_imbalanced <- sepsis.data[-test_indices, ]
  test_df <- sepsis.data[test_indices, ]
  
  cat("Undersampling the training data for this fold...\n")
  sepsis_cases_train <- train_imbalanced %>% filter(SepsisLabel == 1)
  no_sepsis_cases_train <- train_imbalanced %>% filter(SepsisLabel == 0)
  
  if (nrow(sepsis_cases_train) == 0) {
    cat("No positive cases in this training fold. Skipping...\n")
    next
  }
  
  set.seed(42)
  no_sepsis_sampled <- no_sepsis_cases_train %>% sample_n(size = nrow(sepsis_cases_train))
  train_balanced <- bind_rows(sepsis_cases_train, no_sepsis_sampled)
  
  cat(sprintf("Fold %d: Created a balanced training set with %d positive and %d negative samples.\n", 
              i, nrow(sepsis_cases_train), nrow(no_sepsis_sampled)))
  
  train.x <- as.matrix(train_balanced[, setdiff(names(train_balanced), "SepsisLabel"), with = FALSE])
  train.y <- train_balanced$SepsisLabel
  test.x  <- as.matrix(test_df[, setdiff(names(test_df), "SepsisLabel"), with = FALSE])
  test.y  <- test_df$SepsisLabel
  
  dtrain <- xgb.DMatrix(data = train.x, label = train.y)
  dtest <- xgb.DMatrix(data = test.x, label = test.y)
  
  params <- list(
    objective = "binary:logistic", eval_metric = "auc", eta = 0.1, 
    max_depth = 4, subsample = 0.8, colsample_bytree = 0.5
  )
  
  xgb_model <- xgb.train(
    params = params, data = dtrain, nrounds = 500,
    watchlist = list(test = dtest), early_stopping_rounds = 20, verbose = 0
  )
  
  # ---- Evaluate on TEST data ----
  test_preds <- predict(xgb_model, dtest)
  test_results <- calculate_performance_metrics(test.y, test_preds)
  test_results[, Fold := i]
  test_performance_list[[i]] <- test_results
}

# ---- Step 5: Summarize Cross-Validation Results ----
final_test_results <- rbindlist(test_performance_list, fill = TRUE)
metric_cols <- c("Accuracy", "AUC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1_Score")

cat("\n\n================= FINAL XGBoost 5-FOLD CV SUMMARY =================\n")
if (nrow(final_test_results) > 0) {
  cat("\n--- Average TEST Performance Across 5 Folds ---\n")
  avg_test_metrics <- final_test_results[, lapply(.SD, mean, na.rm = TRUE), .SDcols = metric_cols]
  print(round(avg_test_metrics, 4))
  
  cat("\n--- Detailed Test Results Per Fold ---\n")
  print(final_test_results)
} else {
  cat("CV process did not complete for any folds. No results to summarize.\n")
}

# ===================================================================
# End of Script
# ===================================================================
