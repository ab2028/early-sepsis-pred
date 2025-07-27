# ===================================================================
# Sepsis Prediction with Logistic Regression, 5-Fold CV, Imputation, Scaling, and UNDERSAMPLING
# - Manual median imputation -> scaling -> undersampling (50/50) within each fold
# ===================================================================

# ---- Step 0: Load Required Packages ----
library(data.table)
library(caret)
library(dplyr)
library(pROC)
library(PRROC)

set.seed(123)

# ---- Step 1: Load and Prepare Data ----
# source("create_6hr_summary_dataset.R") # change according to dataset
dm <- final_df
dm_model <- subset(dm, select = -c(Patient_ID))

# ---- Step 2: Set up 5-Fold Cross-Validation ----
folds <- createFolds(dm_model$SepsisLabel, k = 5, list = TRUE, returnTrain = FALSE)
test_results_list <- list()

# ---- Step 3: Define Performance Evaluation Function ----
evaluate_model <- function(model, data, cutoff = 0.5) {
  pred <- predict(model, newdata = data, type = "response")
  
  # ROC & AUC (convert to plain numeric)
  roc_obj <- roc(data$SepsisLabel, pred)
  auc_val <- as.numeric(auc(roc_obj))  # << Force numeric here!
  
  pr_obj <- pr.curve(
    scores.class0 = pred[data$SepsisLabel == 1],
    scores.class1 = pred[data$SepsisLabel == 0],
    curve = FALSE
  )
  
  pred_class <- ifelse(pred >= cutoff, 1, 0)
  cm <- confusionMatrix(as.factor(pred_class), as.factor(data$SepsisLabel), positive = "1")
  
  return(data.table(
    Accuracy = as.numeric(cm$overall["Accuracy"]),
    AUC = auc_val,
    AUPRC = pr_obj$auc.integral,
    Sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    Specificity = as.numeric(cm$byClass["Specificity"]),
    Precision = as.numeric(cm$byClass["Precision"]),
    F1_Score = as.numeric(cm$byClass["F1"])
  ))
}

# ---- Step 4: Loop Through Folds for Training, Preprocessing, and Evaluation ----
for (i in 1:length(folds)) {
  cat(sprintf("\n\n================= Processing Fold %d/%d =================\n", i, length(folds)))
  
  test_idx <- folds[[i]]
  train <- dm_model[-test_idx, ]
  test  <- dm_model[test_idx, ]
  
  # --- Step 4.1: Impute Missing Values (Median) ---
  preProc_impute <- preProcess(train, method = "medianImpute")
  train <- predict(preProc_impute, train)
  test  <- predict(preProc_impute, test)
  
  train <- as.data.frame(train)
  test  <- as.data.frame(test)
  
  # --- Step 4.2: Scale Continuous Variables ---
  numeric_cols <- sapply(train, is.numeric)
  binary_cols  <- sapply(train, function(x) length(unique(x)) <= 2)
  scale_cols   <- numeric_cols & !binary_cols & names(train) != "SepsisLabel"
  cols_to_scale <- names(train)[scale_cols]
  
  for (col in cols_to_scale) {
    center_val <- mean(train[[col]], na.rm = TRUE)
    scale_val  <- sd(train[[col]], na.rm = TRUE)
    
    train[[col]] <- (train[[col]] - center_val) / scale_val
    test[[col]]  <- (test[[col]] - center_val) / scale_val
  }
  
  # --- Step 4.3: Undersample Training Data ---
  sepsis_cases <- train %>% filter(SepsisLabel == 1)
  non_sepsis_cases <- train %>% filter(SepsisLabel == 0)
  
  if (nrow(sepsis_cases) == 0) {
    cat("No positive cases in this training fold. Skipping...\n")
    next
  }
  
  set.seed(123)
  non_sepsis_sampled <- non_sepsis_cases %>% sample_n(nrow(sepsis_cases))
  train_balanced <- bind_rows(sepsis_cases, non_sepsis_sampled)
  
  train_balanced <- train_balanced %>% mutate_if(is.character, as.factor)
  train_balanced$SepsisLabel <- factor(train_balanced$SepsisLabel, levels = c(0, 1))
  
  # Ensure test has the same columns as train
  test <- test[, names(train_balanced)]
  
  # --- Step 4.4: Fit Logistic Regression Model ---
  logit_model <- glm(SepsisLabel ~ ., data = train_balanced, family = binomial(link = "logit"))
  
  # --- Step 4.5: Evaluate Model on Test Data ---
  test_results <- evaluate_model(logit_model, test)
  test_results[, Fold := i]
  test_results_list[[i]] <- test_results
  
  cat(sprintf("Fold %d - Test AUC: %.4f | Test AUPRC: %.4f\n", 
              i, test_results$AUC, test_results$AUPRC))
}

# ---- Step 5: Summarize Final Cross-Validation Results ----
final_test_results <- rbindlist(test_results_list, fill = TRUE)

# Force all columns numeric for compatibility in master combine script
for (col in names(final_test_results)) {
  if (col != "Fold") final_test_results[[col]] <- as.numeric(final_test_results[[col]])
}

metric_cols <- c("Accuracy", "AUC", "AUPRC", "Sensitivity", "Specificity", "Precision", "F1_Score")

cat("\n\n================= FINAL Logistic Regression 5-FOLD CV SUMMARY =================\n")
if (nrow(final_test_results) > 0) {
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
