library(ggplot2)
library(gridExtra)

# Vitals
plots_vitals <- lapply(vital_cols, function(v) {
  ggplot(final_df, aes(x = .data[[v]])) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black") +
    labs(title = v, x = "", y = "") +
    theme_minimal() +
    theme(plot.title = element_text(size = 8))
})

# Labs
plots_labs <- lapply(lab_cols, function(v) {
  ggplot(final_df, aes(x = .data[[v]])) +
    geom_histogram(bins = 30, fill = "steelblue", color = "black") +
    labs(title = v, x = "", y = "") +
    theme_minimal() +
    theme(plot.title = element_text(size = 8))
})

# Export vitals figure
pdf("vital_histograms.pdf", width = 10, height = 6)
grid.arrange(grobs = plots_vitals, ncol = 4)
dev.off()

# Export labs figure
pdf("lab_histograms.pdf", width = 10, height = 8)
grid.arrange(grobs = plots_labs, ncol = 4)
dev.off()
