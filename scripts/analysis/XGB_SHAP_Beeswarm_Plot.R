# ===================================================================
# XGBoost Model Training, Evaluation, and SHAP Analysis
# (Train/Test Split with UNDERSAMPLING)
# ===================================================================
# This script:
# 1. Loads engineered ICU dataset (final_df).
# 2. Splits into training (80%) and testing (20%).
# 3. Balances training data using undersampling (50/50).
# 4. Trains an XGBoost model.
# 5. Performs SHAP analysis and plots top features in a beeswarm plot.
# ===================================================================

# Open a plotting device if none exists
#dev.new()


# ---- Step 0: Load Required Packages ----
library(xgboost)
library(data.table)
library(dplyr)
library(caTools)
library(pROC)
library(PRROC)
library(caret)
library(SHAPforxgboost)


# ---- Step 1: Load and Prepare Data ----
# source("create_6hr_summary_dataset.R")  # Uncomment if needed
sepsis.data <- data.table(final_df)

# Drop Patient_ID if present (not a predictor)
if ("Patient_ID" %in% names(sepsis.data)) {
  sepsis.data[, Patient_ID := NULL]
}

# Ensure label column is numeric (0/1)
sepsis.data[, SepsisLabel := as.integer(as.character(SepsisLabel))]


# ---- Step 2: Train/Test Split and UNDERSAMPLING ----
set.seed(123)
sample <- sample.split(sepsis.data$SepsisLabel, SplitRatio = 0.8)
train_imbalanced <- sepsis.data[sample == TRUE, ]
test <- sepsis.data[sample == FALSE, ]

cat("Original training data distribution:\n")
print(table(train_imbalanced$SepsisLabel))

# Balance training set using undersampling
sepsis_cases_train <- train_imbalanced %>% filter(SepsisLabel == 1)
no_sepsis_cases_train <- train_imbalanced %>% filter(SepsisLabel == 0)

if (nrow(sepsis_cases_train) == 0) {
  stop("No positive sepsis cases in training data. Cannot proceed.")
}

set.seed(42)
no_sepsis_sampled <- no_sepsis_cases_train %>% sample_n(size = nrow(sepsis_cases_train))
train_balanced <- bind_rows(sepsis_cases_train, no_sepsis_sampled)

cat("\nBalanced training data distribution:\n")
print(table(train_balanced$SepsisLabel))


# ---- Step 3: Train XGBoost Model ----
train.x <- as.matrix(train_balanced[, setdiff(names(train_balanced), "SepsisLabel"), with = FALSE])
train.y <- train_balanced$SepsisLabel

test.x <- as.matrix(test[, setdiff(names(test), "SepsisLabel"), with = FALSE])
test.y <- test$SepsisLabel

cat("\nTraining XGBoost model...\n")
xgb.model <- xgboost(
  data = train.x,
  label = train.y,
  max.depth = 6,
  eta = 0.1,
  subsample = 0.8,
  colsample_bytree = 0.5,
  nrounds = 500,
  objective = "binary:logistic",
  verbose = 0
)
cat("Model training complete.\n")


# ---- Step 4: SHAP Analysis ----
cat("\n--- SHAP Analysis ---\n")

# Calculate SHAP values and summarize importance
shap_values <- shap.values(xgb_model = xgb.model, X_train = train.x)
shap_summary <- shap_values$mean_shap_score[order(shap_values$mean_shap_score, decreasing = TRUE)]

cat("\nTop 20 features by mean absolute SHAP value:\n")
print(round(head(shap_summary, 20), 5))

# Prepare SHAP values in long format for plotting
shap_long <- shap.prep(xgb_model = xgb.model, X_train = train.x)


# ---- Step 5: Plot SHAP Summary (Top 15 Features) ----
top_n <- 15

# Select top N features based on average SHAP importance
top_features <- shap_long %>%
  group_by(variable) %>%
  summarise(mean_shap = mean(abs(value), na.rm = TRUE)) %>%
  slice_max(mean_shap, n = top_n) %>%
  pull(variable)

# Filter and order data for plotting
shap_long_top <- shap_long %>%
  filter(variable %in% top_features) %>%
  mutate(variable = factor(variable, levels = top_features))

cat(paste0("\nPlotting SHAP summary for top ", top_n, " features...\n"))

# Open plotting device if none exists (needed in some IDEs)
if (!dev.cur()) dev.new(width = 12, height = 8)

# Adjust margins and plot
par(mar = c(5, 12, 2, 2), cex = 1.2)
shap.plot.summary(shap_long_top)


# ===================================================================
# End of Script
# ===================================================================
