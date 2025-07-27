# ---- Load Necessary Libraries ----
library(dplyr)
library(tidyr)


# ---- Step 0: Load and Clean Raw Dataset ----

# runs clean_raw_data.R and creates 'df'
source("clean_raw_data.R")

# ---- Data Leakage Prevention ----
# *NOTE*: This step is critical; labels are shifted back by 6 hours. 

# 1. Calculate true sepsis onset times (shift label forward by 6 hours)
#    If SepsisLabel == 1 at ICULOS t, true onset = t + 6.
sepsis_onset_times <- df %>%
  filter(SepsisLabel == 1) %>%
  group_by(Patient_ID) %>%
  summarise(SepsisOnset_Time = min(ICULOS) + 6, .groups = "drop")

# 2. Ensure all patients are represented (non-septic get Inf)
all_patients <- df %>% distinct(Patient_ID)
sepsis_onset_times <- all_patients %>%
  left_join(sepsis_onset_times, by = "Patient_ID") %>%
  mutate(SepsisOnset_Time = ifelse(is.na(SepsisOnset_Time), Inf, SepsisOnset_Time))

# 3. Filter out all rows at or after the true sepsis onset (leakage prevention)
pre_sepsis_df <- df %>%
  left_join(sepsis_onset_times, by = "Patient_ID") %>%
  filter(ICULOS < SepsisOnset_Time)

# ---- Baseline Feature Extraction ----

# 4. Create the final patient-level outcome label (did the patient ever develop sepsis?)
sepsis_outcome_label <- df %>%
  group_by(Patient_ID) %>%
  summarise(SepsisLabel_final = as.integer(any(SepsisLabel == 1)), .groups = "drop")

# 5. For each patient, calculate the first available (non-NA) value of each variable 
#    using the leakage-free dataset (pre_sepsis_df).
baseline_df <- pre_sepsis_df %>%
  select(-SepsisLabel, -SepsisOnset_Time) %>%
  group_by(Patient_ID) %>%
  summarise(across(everything(), ~ .x[which(!is.na(.x))[1]]), .groups = "drop")

# ---- Combine Features and Labels ----

# 6. Join baseline features with outcome labels; retain patients without pre-sepsis data
final_df <- full_join(sepsis_outcome_label, baseline_df, by = "Patient_ID") %>%
  rename(SepsisLabel = SepsisLabel_final)

# 7. Drop columns that are not required
final_df <- final_df[, !(names(final_df) %in% c("ICULOS"))]

# Optionally drop any rows with missing values (if listwise deletion)
final_df <- na.omit(final_df)

# ---- Validate and Save ----
# Print label distribution
table(final_df$SepsisLabel)

# Optionally, save to CSV
# write.csv(final_df, "baseline_data_leakage_free.csv", row.names = FALSE)

                   