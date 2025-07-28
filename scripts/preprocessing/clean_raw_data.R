library(data.table)
library(dplyr)

#Load raw dataset
df <- fread("data/raw/rawdataset.csv")

#Remove unnecessary or unused columns
remove_cols <- c("V1", "Hour", "HospAdmTime", "EtCO2", "Bilirubin_direct", "TroponinI", "Fibrinogen", "Hct",    "BaseExcess", "AST", "Alkalinephos", "Calcium", "Chloride",
                 "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "PTT", "SaO2")
df[, (remove_cols) := NULL]

#Correct improperly recorded FiO2 values
if ("FiO2" %in% names(df)) {
  while (any(df$FiO2 > 1, na.rm = TRUE)) {
    df[FiO2 > 1, FiO2 := FiO2 / 10]
  }
}

#Merge Unit1 and Unit2 in to a single unit variable
df$Unit1 <- as.numeric(as.character(df$Unit1))
df$Unit2 <- as.numeric(as.character(df$Unit2))
df$Unit1[is.na(df$Unit1)] <- 0
df$Unit2[is.na(df$Unit2)] <- 0
df$Unit <- ifelse(df$Unit1 == 1, 1, ifelse(df$Unit2 == 1, 2, 0))
#Drop original Unit1 and Unit2 Columns
df <- df %>% select(-Unit1, -Unit2)
