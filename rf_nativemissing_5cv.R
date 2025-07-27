# ===================================================================
# Sepsis Prediction with Ranger (Random Forest), 5-Fold CV
# - Native handling of missing values in predictors (no imputation)
# - Manual undersampling (50/50) within each fold
# ===================================================================

# ---- Step 1: Load Required Packages ----
library(ranger)       # Fast Random Forest implementation
library(data.table)   # Efficient data manipulation
library(dplyr)        # Data wrangling functions (filter, bind_rows, etc.)
library(pROC)         # ROC and AUC calculation
library(PRROC)        # Precision-Recall curves and AUPRC calculation
library(caret)        # For createFolds and confusionMatrix

# ---- Step 2: Load and Prepare Data ----
#source("create_6hr_summary_dataset.R") #change according to which dataset is used

sepsis.data <- data.table(final_df)
sepsis.data <- sepsis.data[!is.na(SepsisLabel)]   # Remove rows with missing labels

if ("Patient_ID" %in% names(sepsis.data)) {
  sepsis.data[, Patient_ID := NULL]   # Remove Patient_ID column if present
}

# ---- Step 3: Performance Calculation Function ----
calculate_performance_metrics <- function(model, data, true_labels) {
  # Ranger requires predictors only for prediction
  predictor_data <- data[, setdiff(names(data), "SepsisLabel"), with = FALSE]
  pred_obj <- predict(model, data = predictor_data)
  pred_probs <- pred_obj$predictions[, "1"] # Probabilities for class '1'
  
  # Threshold-independent metrics
  auc_score <- as.numeric(pROC::auc(true_labels, pred_probs, quiet = TRUE))
  pr_obj <- pr.curve(scores.class0 = pred_probs[true_labels == 1], 
                     scores.class1 = pred_probs[true_labels == 0], 
                     curve = FALSE)
  auprc_score <- pr_obj$auc.integral
  
  # Threshold-dependent metrics
  pred_labels <- factor(ifelse(pred_probs > 0.5, 1, 0), levels = c(0, 1))
  cm_result <- try(confusionMatrix(data = pred_labels, reference = true_labels, positive = "1"), silent = TRUE)
  
  if (inherits(cm_result, "try-error")) {
    accuracy <- NA; sensitivity <- NA; specificity <- NA
    precision <- NA; f1 <- NA
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

# ---- Step 4: Set up 5-Fold Cross-Validation ----
set.seed(123)
sepsis.data[, SepsisLabel := factor(SepsisLabel, levels = c(0, 1))]
folds <- createFolds(sepsis.data$SepsisLabel, k = 5, list = TRUE, returnTrain = FALSE)

test_performance_list <- list()

# ---- Step 5: Training, Undersampling, and Evaluation Loop ----
for (i in 1:length(folds)) {
  cat(sprintf("\n\n================= Processing Fold %d/%d =================\n", i, length(folds)))
  
  test_indices <- folds[[i]]
  train_imbalanced <- sepsis.data[-test_indices, ]
  test_df <- sepsis.data[test_indices, ]
  
  # --- Undersample training set ---
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
  
  cat(sprintf("Fold %d: Created a balanced training set with %d samples.\n", i, nrow(train_balanced)))
  
  # --- Train Ranger model ---
  rf_model <- ranger(
    formula = SepsisLabel ~ .,
    data = train_balanced,
    num.trees = 500,
    probability = TRUE,      # Required for probability predictions
    importance = 'impurity', # Gini importance
    max.depth = 10           # Regularization (optional)
  )
  
  # --- Evaluate on TEST data only ---
  test_results <- calculate_performance_metrics(rf_model, test_df, test_df$SepsisLabel)
  test_results[, Fold := i]
  test_performance_list[[i]] <- test_results
  
  cat(sprintf("Fold %d - Test AUC: %.4f\n", i, test_results$AUC))
}

# ---- Step 6: Summarize Cross-Validation Results ----
final_test_results <- rbindlist(test_performance_list, fill = TRUE)
metric_cols <- c("Accuracy", "AUC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1_Score")

cat("\n\n================= FINAL Ranger 5-FOLD CV SUMMARY =================\n")

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
