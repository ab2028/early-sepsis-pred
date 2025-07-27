library(data.table)
library(dplyr)

# ---- Step 0: Load and Clean Raw Dataset ----

# runs clean_raw_data.R and creates 'df'
source("clean_raw_data.R")

# ---- Step 1: Filter for Eligible Population ----

OBSERVATION_WINDOW_HOURS <- 24
PREDICTION_START_HOUR <- OBSERVATION_WINDOW_HOURS + 1   # hour 25 onward

# 1.1: Keep patients with at least 25 hours of data (so the full prediction horizon is observed)
patient_los <- df[, .(LOS = max(ICULOS)), by = Patient_ID]
eligible_los_ids <- patient_los[LOS >= PREDICTION_START_HOUR, Patient_ID]
df_filtered <- df[Patient_ID %in% eligible_los_ids, ]

# 1.2: Exclude patients who developed sepsis within the first 18 hours (shifted labels would make these ambiguous)
sepsis_onset_window <- df_filtered[ICULOS >= 0 & ICULOS <= (OBSERVATION_WINDOW_HOURS - 6)]
early_sepsis_ids <- sepsis_onset_window[SepsisLabel == 1, unique(Patient_ID)]

# 1.3: Final cohort excludes early-onset septic patients
df_cohort <- df_filtered[!Patient_ID %in% early_sepsis_ids, ]

# 1.4: Restrict the data to the first 24-hour observation window for feature calculation
df_window <- df_cohort[ICULOS <= OBSERVATION_WINDOW_HOURS, ]


# ---- Step 2: Extract Features (Modular) ----

# Measurement columns to include (excluding identifiers, demographics, outcome, and dropped labs)
meas_cols <- setdiff(colnames(df_window),
                     c("Patient_ID", "ICULOS", "SepsisLabel", "Gender", "Age", "Unit"))

# Ensure chronological order per patient
setorder(df_window, Patient_ID, ICULOS)

# Modular toggles: which features to compute per measurement
include_baseline    <- TRUE   # First recorded value
include_count       <- TRUE   # Number of measurements
include_last_first  <- TRUE   # Last - first value
include_range       <- TRUE   # Max - min

# Extract features per patient and measurement column
features_list <- df_window[, {
  res <- list()
  for (col in meas_cols) {
    x <- .SD[[col]]
    valid_x <- x[!is.na(x)]
    
    if (length(valid_x) > 0) {
      if (include_baseline)    res[[paste0(col, "_base")]]  <- valid_x[1]
      if (include_count)       res[[paste0(col, "_count")]] <- length(valid_x)
      if (include_last_first)  res[[paste0(col, "_delta")]] <- valid_x[length(valid_x)] - valid_x[1]
      if (include_range)       res[[paste0(col, "_range")]] <- max(valid_x) - min(valid_x)
    } else {
      if (include_baseline)    res[[paste0(col, "_base")]]  <- NA_real_
      if (include_count)       res[[paste0(col, "_count")]] <- 0L
      if (include_last_first)  res[[paste0(col, "_delta")]] <- NA_real_
      if (include_range)       res[[paste0(col, "_range")]] <- NA_real_
    }
  }
  res
}, by = Patient_ID]


# ---- Step 3: Add Demographics and Final Label ----

# Static demographic info per patient (Age, Gender, Unit)
demographics <- df_window[, .(
  Age    = first(Age),
  Gender = first(Gender),
  Unit   = first(Unit)
), by = Patient_ID]

# Final outcome label: did the patient ever develop sepsis?
label_dt <- df_cohort[, .(
  SepsisLabel = as.integer(any(SepsisLabel == 1))
), by = Patient_ID]

# Merge features, demographics, and labels into the final dataset
final_df <- Reduce(function(x, y) merge(x, y, by = "Patient_ID", all.x = TRUE),
                   list(features_list, demographics, label_dt))

# Optional: Drop additional columns flagged for exclusion (if needed)

# Result: `final_df` contains one row per patient with engineered features and labels
