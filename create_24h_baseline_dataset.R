library(data.table)
library(dplyr)

# ---- Step 0: Load and Clean Raw Dataset ----

# runs clean_raw_data.R and creates 'df'
source("clean_raw_data.R")  

# ---- Step 1: Filter for Eligible Population ----
OBSERVATION_WINDOW_HOURS <- 24
PREDICTION_START_HOUR <- OBSERVATION_WINDOW_HOURS + 1   # hour 25 onward

# 1.1: Keep patients with at least 25 hours of data
patient_los <- df[, .(LOS = max(ICULOS)), by = Patient_ID]
eligible_los_ids <- patient_los[LOS >= PREDICTION_START_HOUR, Patient_ID]
df_filtered <- df[Patient_ID %in% eligible_los_ids, ]

# 1.2: Identify patients with sepsis onset within the first 24 hours (ICULOS ≤ 18)
sepsis_onset_window <- df_filtered[ICULOS <= (OBSERVATION_WINDOW_HOURS - 6)]
early_sepsis_ids <- sepsis_onset_window[SepsisLabel == 1, unique(Patient_ID)]

# 1.3: Exclude early sepsis patients from the cohort
df_cohort <- df_filtered[!Patient_ID %in% early_sepsis_ids, ]

# 1.4: Limit data to the first 24-hour observation window for feature calculation
df_window <- df_cohort[ICULOS <= OBSERVATION_WINDOW_HOURS, ]


# ---- Step 2: Extract Baseline Features Only ----

# Measurement columns (exclude IDs, time, demographics, and outcome)
meas_cols <- setdiff(colnames(df_window), 
                     c("Patient_ID", "ICULOS", "SepsisLabel", "Gender", "Age", "Unit"))

# Sort by patient and time to get the correct "first" observation
setorder(df_window, Patient_ID, ICULOS)

# For each measurement column, get the first available non-NA value per patient
features_list <- df_window[, lapply(.SD, function(x) x[!is.na(x)][1]), 
                           by = Patient_ID, 
                           .SDcols = meas_cols]


# ---- Step 3: Add Demographics and Final Sepsis Label ----

# 3.1: Demographics (age, gender, unit) are static; take the first available value
demographics <- df_window[, .(
  Age = first(Age),
  Gender = first(Gender),
  Unit = first(Unit)
), by = Patient_ID]

# 3.2: Final outcome label (1 = sepsis occurred at any time)
label_dt <- df_cohort[, .(SepsisLabel = as.integer(any(SepsisLabel == 1))), 
                      by = Patient_ID]

# 3.3: Merge features, demographics, and labels into final dataset
final_df <- Reduce(function(x, y) merge(x, y, by = "Patient_ID", all.x = TRUE),
                   list(features_list, demographics, label_dt))
