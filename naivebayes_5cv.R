# ===================================================================
# Sepsis Prediction with Naive Bayes, 5-Fold CV, and UNDERSAMPLING
# - Manual median imputation BEFORE undersampling (50/50) within each fold
# ===================================================================

# ---- Step 0: Load Required Packages ----
library(data.table)
library(dplyr)
library(caret)
library(pROC)
library(PRROC)
library(e1071)

# ---- Step 2: Load and Prepare Data ----
#source("create_6hr_summary_dataset.R") #change according to which dataset is used

sepsis.data <- data.table(final_df)
sepsis.data <- sepsis.data[!is.na(SepsisLabel)]   # Remove rows with missing labels

if ("Patient_ID" %in% names(sepsis.data)) {
  sepsis.data[, Patient_ID := NULL]   # Remove Patient_ID column if present
}

# Convert label to factor
sepsis.data[, SepsisLabel := factor(SepsisLabel, levels = c(0, 1))]

# ---- Step 3: Performance Evaluation Function ----
calculate_performance_metrics <- function(model, data_to_eval, true_labels) {
  pred_probs <- predict(model, newdata = data_to_eval, type = "raw")[, "1"]
  pred_labels <- factor(ifelse(pred_probs > 0.5, "1", "0"), levels = c("0", "1"))
  
  valid_indices <- !is.na(true_labels) & !is.na(pred_probs)
  clean_labels <- true_labels[valid_indices]
  clean_probs <- pred_probs[valid_indices]
  clean_preds <- pred_labels[valid_indices]
  
  if (length(unique(clean_labels)) < 2) {
    warning("Not enough classes to calculate metrics.")
    return(data.table(Accuracy=NA, AUC=NA, AUPRC=NA, Sensitivity=NA, Specificity=NA, Precision=NA, F1_Score=NA))
  }
  
  auc_score <- as.numeric(pROC::auc(clean_labels, clean_probs, quiet = TRUE))
  
  scores_class_1 <- clean_probs[clean_labels == "1"]
  scores_class_0 <- clean_probs[clean_labels == "0"]
  
  auprc_score <- if(length(scores_class_1) > 0 && length(scores_class_0) > 0) {
    pr_obj <- pr.curve(scores.class0 = scores_class_1, scores.class1 = scores_class_0, curve = FALSE)
    pr_obj$auc.integral
  } else {
    NA
  }
  
  cm_result <- caret::confusionMatrix(data = clean_preds, reference = clean_labels, positive = "1")
  
  return(data.table(
    Accuracy = cm_result$overall['Accuracy'],
    AUC = auc_score,
    AUPRC = auprc_score,
    Sensitivity = cm_result$byClass['Sensitivity'],
    Specificity = cm_result$byClass['Specificity'],
    Precision = cm_result$byClass['Pos Pred Value'],
    F1_Score = cm_result$byClass['F1']
  ))
}

# ---- Step 4: Set up 5-Fold Cross-Validation ----
set.seed(123)
folds <- createFolds(sepsis.data$SepsisLabel, k = 5, list = TRUE, returnTrain = FALSE)
test_performance_list <- list()

# ---- Step 5: Training Loop (Imputation -> Undersampling -> Evaluation) ----
for (i in 1:length(folds)) {
  cat(sprintf("\n\n================= Processing Fold %d/%d =================\n", i, length(folds)))
  
  test_indices <- folds[[i]]
  train_imbalanced <- sepsis.data[-test_indices, ]
  test_df <- sepsis.data[test_indices, ]
  
  # --- Step 5.1: Median Imputation ---
  predictor_cols <- setdiff(names(train_imbalanced), "SepsisLabel")
  for (col in predictor_cols) {
    if (any(is.na(train_imbalanced[[col]]))) {
      median_val <- median(train_imbalanced[[col]], na.rm = TRUE)
      set(train_imbalanced, i = which(is.na(train_imbalanced[[col]])), j = col, value = median_val)
      set(test_df, i = which(is.na(test_df[[col]])), j = col, value = median_val)
    }
  }
  
  # --- Step 5.2: Undersampling ---
  sepsis_cases_train <- train_imbalanced[SepsisLabel == "1"]
  no_sepsis_cases_train <- train_imbalanced[SepsisLabel == "0"]
  
  if (nrow(sepsis_cases_train) == 0) {
    cat("No positive cases in this training fold. Skipping...\n")
    next
  }
  
  set.seed(42)
  no_sepsis_sampled <- no_sepsis_cases_train[sample(.N, size = nrow(sepsis_cases_train))]
  train_balanced <- rbind(sepsis_cases_train, no_sepsis_sampled)
  
  cat(sprintf("Training data balanced: %d Sepsis, %d No Sepsis\n", 
              sum(train_balanced$SepsisLabel == 1), 
              sum(train_balanced$SepsisLabel == 0)))
  
  # --- Step 5.3: Train Naive Bayes model ---
  nb_model <- naiveBayes(SepsisLabel ~ ., data = train_balanced)
  
  # --- Step 5.4: Evaluate only on TEST data ---
  test_results <- calculate_performance_metrics(nb_model, test_df, test_df$SepsisLabel)
  test_results[, Fold := i]
  test_performance_list[[i]] <- test_results
  
  cat(sprintf("Fold %d - Test AUC: %.4f\n", i, test_results$AUC))
}

# ---- Step 6: Summarize Results ----
final_test_results <- rbindlist(test_performance_list, fill = TRUE)
metric_cols <- c("Accuracy", "AUC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1_Score")

cat("\n\n================= FINAL Naive Bayes 5-FOLD CV SUMMARY =================\n")
if (nrow(final_test_results) > 0) {
  cat("\n--- Average TEST Performance Across 5 Folds ---\n")
  avg_test_metrics <- final_test_results[, lapply(.SD, mean, na.rm = TRUE), .SDcols = metric_cols]
  print(round(avg_test_metrics, 4))
} else {
  cat("CV process did not complete for any folds. No results to summarize.\n")
}

# ===================================================================
# End of Script
# ===================================================================
