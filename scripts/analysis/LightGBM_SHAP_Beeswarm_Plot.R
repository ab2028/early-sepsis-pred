# ===================================================================
# LightGBM Model Training, Evaluation, and SHAP Analysis 
# (Train/Test Split with UNDERSAMPLING)
# ===================================================================
# This script:
# 1. Loads engineered ICU dataset (final_df).
# 2. Splits the dataset into training (80%) and validation (20%).
# 3. Balances the training data using undersampling (50/50).
# 4. Trains a LightGBM model.
# 5. Performs SHAP analysis and plots top features in a beeswarm plot.
# ===================================================================


# ---- Step 0: Load Required Packages ----
library(data.table)
library(dplyr)
library(lightgbm)
library(pROC)
library(PRROC)
library(caTools)
library(SHAPforxgboost)  # For SHAP values (works with tree-based models)


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
train_df <- sepsis.data[sample == TRUE, ]
valid_df <- sepsis.data[sample == FALSE, ]

cat("Original training data distribution:\n")
print(table(train_df$SepsisLabel))
cat("\nValidation data distribution:\n")
print(table(valid_df$SepsisLabel))

# Balance training data using undersampling
sepsis_cases <- train_df %>% filter(SepsisLabel == 1)
no_sepsis_cases <- train_df %>% filter(SepsisLabel == 0)

set.seed(42)
no_sepsis_sampled <- no_sepsis_cases %>% sample_n(size = nrow(sepsis_cases))
train_data_balanced <- bind_rows(sepsis_cases, no_sepsis_sampled)

cat("\nBalanced training data distribution:\n")
print(table(train_data_balanced$SepsisLabel))


# ---- Step 3: Prepare Data for LightGBM ----
predictor_cols <- setdiff(names(train_data_balanced), "SepsisLabel")

x_train <- as.matrix(train_data_balanced[, ..predictor_cols])
y_train <- train_data_balanced$SepsisLabel

x_valid <- as.matrix(valid_df[, ..predictor_cols])
y_valid <- valid_df$SepsisLabel

dtrain <- lgb.Dataset(data = x_train, label = y_train)
dvalid <- lgb.Dataset(data = x_valid, label = y_valid, reference = dtrain)


# ---- Step 4: Train LightGBM Model ----
params <- list(
  objective = "binary",
  metric = "auc",
  boosting = "gbdt",
  num_leaves = 11,
  max_depth = 5,
  lambda_l2 = 0.5,
  lambda_l1 = 0.1,
  min_child_samples = 50,
  learning_rate = 0.01,
  verbosity = -1
)

model <- lgb.train(
  params = params,
  data = dtrain,
  valids = list(validation = dvalid),
  nrounds = 500,
  early_stopping_rounds = 50,
  verbose = 0
)


# ---- Step 5: Feature Importance ----
feature_imp <- lgb.importance(model = model)
cat("\n--- Top 30 Most Important Features ---\n")
print(head(feature_imp, 30))
# Optionally plot: lgb.plot.importance(tree_imp = feature_imp, top_n = 15, measure = "Gain")


# ---- Step 6: SHAP Analysis ----
cat("\n--- SHAP Analysis ---\n")

# 1. Compute SHAP values on training data
shap_values <- shap.values(xgb_model = model, X_train = x_train)

# 2. Summarize mean absolute SHAP values
shap_summary <- shap_values$mean_shap_score
shap_summary <- shap_summary[order(shap_summary, decreasing = TRUE)]
cat("\nTop 20 features by mean absolute SHAP value:\n")
print(round(head(shap_summary, 20), 5))

# 3. Prepare SHAP values for plotting
shap_long <- shap.prep(xgb_model = model, X_train = x_train)

# 4. Filter to Top-N features for plotting
top_n <- 15
top_features <- shap_long %>%
  group_by(variable) %>%
  summarise(mean_shap = mean(abs(value), na.rm = TRUE)) %>%
  slice_max(mean_shap, n = top_n) %>%
  arrange(desc(mean_shap)) %>%
  pull(variable)

shap_long_top <- shap_long %>%
  filter(variable %in% top_features) %>%
  mutate(variable = factor(variable, levels = top_features))


# ---- Step 7: Plot SHAP Summary ----
cat(paste0("\nPlotting SHAP summary for top ", top_n, " features...\n"))

# Open plotting device if none exists (prevents silent failures)
if (!dev.cur()) dev.new(width = 12, height = 8)

par(mar = c(5, 12, 2, 2), cex = 1.2)
shap.plot.summary(shap_long_top)


# ===================================================================
# End of Script
# ===================================================================
