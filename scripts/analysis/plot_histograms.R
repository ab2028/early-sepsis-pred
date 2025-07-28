# Load necessary libraries
library(ggplot2)
library(gridExtra)

# Assume 'final_df' is your pre-existing data frame

# Define column groups
vital_cols <- c("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp")
lab_cols   <- c("FiO2", "Platelets", "Hgb", "WBC", "Lactate", 
                "Creatinine", "BUN", "Glucose", "pH", "PaCO2", "HCO3")
dem_cols   <- c("Age", "Gender", "Unit")

# --- Vitals (Continuous) ---
plots_vitals <- lapply(vital_cols, function(v) {
  ggplot(final_df, aes(x = .data[[v]])) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black", na.rm = TRUE) +
    labs(title = v, x = NULL, y = NULL) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 10), # Centered title
          axis.text.x = element_text(angle = 45, hjust = 1)) # Angled labels for readability
})

# --- Labs (Continuous) ---
plots_labs <- lapply(lab_cols, function(v) {
  ggplot(final_df, aes(x = .data[[v]])) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black", na.rm = TRUE) +
    labs(title = v, x = NULL, y = NULL) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 10),
          axis.text.x = element_text(angle = 45, hjust = 1))
})

# --- Demographics (Mixed: Continuous and Categorical) ---
plots_demographics <- lapply(dem_cols, function(v) {
  p <- ggplot(final_df, aes(x = .data[[v]]))
  
  # Use geom_bar for categorical/factor variables, geom_histogram for numeric
  if (is.factor(final_df[[v]]) || is.character(final_df[[v]])) {
    # For Gender and Unit
    p <- p + geom_bar(fill = "skyblue", color = "black", na.rm = TRUE)
  } else {
    # For Age
    p <- p + geom_histogram(bins = 30, fill = "steelblue", color = "black", na.rm = TRUE)
  }
  
  p + labs(title = v, x = NULL, y = NULL) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 10))
})


# --- Export Figures ---

# Export vitals figure (7 plots, maybe 4x2 layout)
pdf("vital_histograms.pdf", width = 10, height = 5)
grid.arrange(grobs = plots_vitals, ncol = 4)
dev.off()

# Export labs figure (11 plots, maybe 4x3 layout)
pdf("lab_histograms.pdf", width = 10, height = 8)
grid.arrange(grobs = plots_labs, ncol = 4)
dev.off()

# Export demographics figure (3 plots, 3x1 layout)
pdf("demographic_plots.pdf", width = 10, height = 3.5)
grid.arrange(grobs = plots_demographics, ncol = 3)
dev.off()

