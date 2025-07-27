# ===================================================================
# Sepsis Prediction with SVM, 5-Fold CV, Imputation, Scaling, and UNDERSAMPLING
# - Manual median imputation -> scaling -> undersampling (50/50) within each fold
# ===================================================================

# ---- Step 0: Load Required Packages ----
library(data.table)
library(dplyr)
library(caret)
library(pROC)
library(PRROC)
library(e1071)

# ---- Step 1: Load and Prepare Data ----
#source("create_6hr_summary_dataset.R") # change according to which dataset is used

sepsis.data <- data.table(final_df)
sepsis.data <- sepsis.data[!is.na(SepsisLabel)]   # Remove rows with missing labels

if ("Patient_ID" %in% names(sepsis.data)) {
  sepsis.data[, Patient_ID := NULL]   # Remove Patient_ID column if present
}
sepsis.data[, SepsisLabel := factor(SepsisLabel, levels = c(0, 1))]

# ---- Step 2: Define Performance Function ----
calculate_svm_performance <- function(model, data_x, true_labels) {
  pred_labels <- predict(model, newdata = data_x)
  pred_probs_attr <- attr(predict(model, newdata = data_x, probability = TRUE), "probabilities")
  pred_probs <- pred_probs_attr[, "1"]
  
  valid_indices <- !is.na(true_labels) & !is.na(pred_probs)
  clean_labels <- true_labels[valid_indices]
  clean_probs <- pred_probs[valid_indices]
  clean_preds <- pred_labels[valid_indices]
  
  if (length(unique(clean_labels)) < 2) {
    warning("Not enough classes to calculate metrics.")
    return(data.table(Accuracy=NA, AUC=NA, AUPRC=NA, Sensitivity=NA, Specificity=NA, Precision=NA, F1_Score=NA))
  }
  
  auc_score <- as.numeric(pROC::auc(clean_labels, clean_probs, quiet = TRUE))
  
  scores_class_1 <- clean_probs[clean_labels == 1]
  scores_class_0 <- clean_probs[clean_labels == 0]
  
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

# ---- Step 3: Set up 5-Fold Cross-Validation ----
set.seed(123)
folds <- createFolds(sepsis.data$SepsisLabel, k = 5, list = TRUE, returnTrain = FALSE)

# Define kernels to evaluate (can extend this list)
svm_kernels <- c("linear", "polynomial", "radial", "sigmoid")

# Storage for results
all_results <- sapply(svm_kernels, function(k) list(test=list()), simplify = FALSE)

# ---- Step 4: Loop Through Folds for Training and Evaluation ----
for (i in 1:length(folds)) {
  cat(sprintf("\n\n================= Processing Fold %d/%d =================\n", i, length(folds)))
  
  test_indices <- folds[[i]]
  train_imbalanced <- sepsis.data[-test_indices, ]
  test_df <- sepsis.data[test_indices, ]
  
  # ---- Step 4.1: Impute Missing Values ----
  predictor_cols <- setdiff(names(train_imbalanced), "SepsisLabel")
  for (col in predictor_cols) {
    if (any(is.na(train_imbalanced[[col]]))) {
      median_val <- median(train_imbalanced[[col]], na.rm = TRUE)
      set(train_imbalanced, i = which(is.na(train_imbalanced[[col]])), j = col, value = median_val)
      set(test_df, i = which(is.na(test_df[[col]])), j = col, value = median_val)
    }
  }
  
  # ---- Step 4.2: Scale Data ----
  preproc_params <- preProcess(train_imbalanced[, ..predictor_cols], method = c("center", "scale"))
  train_imputed_scaled <- predict(preproc_params, train_imbalanced)
  test_imputed_scaled <- predict(preproc_params, test_df)
  
  # ---- Step 4.3: Undersample ----
  sepsis_cases_train <- train_imputed_scaled[SepsisLabel == "1"]
  no_sepsis_cases_train <- train_imputed_scaled[SepsisLabel == "0"]
  
  if (nrow(sepsis_cases_train) == 0) {
    cat("No positive cases in this training fold. Skipping...\n")
    next
  }
  
  set.seed(42)
  no_sepsis_sampled <- no_sepsis_cases_train[sample(.N, size = nrow(sepsis_cases_train))]
  train_balanced_scaled <- rbind(sepsis_cases_train, no_sepsis_sampled)
  
  train.x <- train_balanced_scaled[, ..predictor_cols]
  train.y <- train_balanced_scaled$SepsisLabel
  test.x  <- test_imputed_scaled[, ..predictor_cols]
  test.y  <- test_imputed_scaled$SepsisLabel
  
  # ---- Step 4.4: Train and Evaluate per Kernel ----
  for (kernel_type in svm_kernels) {
    cat(sprintf("\n--- Training SVM with %s kernel ---\n", kernel_type))
    
    svm_model <- svm(
      x = train.x, y = train.y,
      kernel = kernel_type,
      probability = TRUE
    )
    
    # Evaluate on TEST set only
    test_results <- calculate_svm_performance(svm_model, test.x, test.y)
    test_results[, Fold := i]
    all_results[[kernel_type]]$test[[i]] <- test_results
    
    # --- Create special variable if kernel is radial ---
    if (kernel_type == "radial") {
      if (!exists("final_radial_test_results")) {
        final_radial_test_results <- test_results
      } else {
        final_radial_test_results <- rbind(final_radial_test_results, test_results, fill = TRUE)
      }
    }
    
    cat(sprintf("Fold %d (%s) - Test AUC: %.4f\n", 
                i, kernel_type, test_results$AUC))
  }
}

# ---- Step 5: Summarize Cross-Validation Results ----
metric_cols <- c("Accuracy", "AUC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1_Score")

for (kernel_type in svm_kernels) {
  cat(sprintf("\n\n================= FINAL SVM (%s KERNEL) 5-FOLD CV SUMMARY =================\n", toupper(kernel_type)))
  
  final_test_results <- rbindlist(all_results[[kernel_type]]$test, fill = TRUE)
  
  if (nrow(final_test_results) > 0) {
    cat("\n--- Average TEST Performance Across 5 Folds ---\n")
    avg_test_metrics <- final_test_results[, lapply(.SD, mean, na.rm = TRUE), .SDcols = metric_cols]
    print(round(avg_test_metrics, 4))
  } else {
    cat("CV process did not complete for this kernel. No results to summarize.\n")
  }
}

# ===================================================================
# End of Script
# ===================================================================
