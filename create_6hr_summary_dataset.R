library(data.table)
library(dplyr)

# ---- Step 0: Load and Clean Raw Dataset ----

# runs clean_raw_data.R and creates 'df'
source("clean_raw_data.R")  

# ---- Step 1: Define Windows and Identify Patient Cohorts ----
FEATURE_WINDOW_HOURS <- 6
PREDICTION_WINDOW_HOURS <- 24
PREDICTION_START_HOUR <- FEATURE_WINDOW_HOURS + 1  # hour 7 onward

# Identify septic and non-septic patients
all_septic_ids <- df[SepsisLabel == 1, unique(Patient_ID)]
patient_los <- df[, .(LOS = max(ICULOS)), by = Patient_ID]

# Keep non-septic patients with sufficient LOS
non_septic_ids <- patient_los[
  !Patient_ID %in% all_septic_ids & LOS >= PREDICTION_WINDOW_HOURS,
  Patient_ID
]

# Final cohort
final_cohort_ids <- c(all_septic_ids, non_septic_ids)
df_cohort <- df[Patient_ID %in% final_cohort_ids, ]

# Label patients as positive/negative
sepsis_onset_flags <- df_cohort[SepsisLabel == 1, .(FirstFlagTime = min(ICULOS)), by = Patient_ID]
positive_patient_ids <- sepsis_onset_flags[
  FirstFlagTime >= (PREDICTION_START_HOUR - 6) &
    FirstFlagTime <= (PREDICTION_WINDOW_HOURS - 6),
  Patient_ID
]

label_dt <- data.table(Patient_ID = final_cohort_ids)
label_dt[, SepsisLabel := ifelse(Patient_ID %in% positive_patient_ids, 1, 0)]


# ---- Step 2: Extract Features from the First 6 Hours ----
df_window <- df_cohort[ICULOS <= FEATURE_WINDOW_HOURS, ]
setorder(df_window, Patient_ID, ICULOS)

# Separate VITALS and LABS
vital_cols <- c("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp")
lab_cols   <- c("FiO2", "Platelets", "Hgb", "WBC", "Lactate", 
                "Creatinine", "BUN", "Glucose", "pH", "PaCO2", "HCO3")

# Toggle summary stats for vitals
include_base_vital   <- TRUE
include_count_vital  <- TRUE
include_delta_vital  <- TRUE
include_sd_vital     <- TRUE

# Toggle summary stats for labs
include_base_lab     <- TRUE
include_count_lab    <- TRUE
include_delta_lab    <- FALSE
include_sd_lab       <- FALSE

# Generalized feature extraction function
calculate_features <- function(x, is_vital) {
  valid_x <- x[!is.na(x)]
  
  base_val  <- if ((is_vital && include_base_vital) || (!is_vital && include_base_lab)) {
    ifelse(length(valid_x) > 0, valid_x[1], NA_real_)
  } else NULL
  
  count_val <- if ((is_vital && include_count_vital) || (!is_vital && include_count_lab)) {
    sum(!is.na(x))
  } else NULL
  
  delta_val <- if ((is_vital && include_delta_vital) || (!is_vital && include_delta_lab)) {
    if (length(valid_x) > 0) tail(valid_x, 1) - valid_x[1] else NA_real_
  } else NULL
  
  sd_val    <- if ((is_vital && include_sd_vital) || (!is_vital && include_sd_lab)) {
    if (length(valid_x) > 1) sd(valid_x) else NA_real_
  } else NULL
  
  out <- list()
  if (!is.null(base_val))  out$base <- base_val
  if (!is.null(count_val)) out$count <- count_val
  if (!is.null(delta_val)) out$last_minus_first <- delta_val
  if (!is.null(sd_val))    out$sd <- sd_val
  
  return(out)
}

# Compute features per patient
features_list <- df_window[, {
  v_res <- lapply(.SD[, vital_cols, with = FALSE], function(col) calculate_features(col, TRUE))
  l_res <- lapply(.SD[, lab_cols, with = FALSE],   function(col) calculate_features(col, FALSE))
  unlist(c(v_res, l_res), recursive = FALSE)
}, by = Patient_ID]

# Clean feature names
setnames(features_list, gsub("\\.", "_", names(features_list)))


# ---- Step 3: Add Demographics and Merge into Final DataFrame ----
demographics <- df_window[, .(Age = first(Age), Gender = first(Gender), Unit = first(Unit)), by = Patient_ID]

# Merge features, demographics, and labels
final_df <- Reduce(function(x, y) merge(x, y, by = "Patient_ID", all.x = TRUE),
                   list(features_list, demographics, label_dt))
final_df <- final_df[Patient_ID %in% features_list$Patient_ID]

# Output final counts
cat("Final counts for the modeling task:\n")
print(table(final_df$SepsisLabel, dnn = "Sepsis Status (0=Non-Septic, 1=Septic)"))
