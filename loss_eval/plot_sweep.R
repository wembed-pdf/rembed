#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(tidyr))

# Read unified sweep results
data <- read.csv("sweep.csv")

# Split into LSH and LossyQuery data
lsh_data <- data %>% filter(sweep_type == "LSH")
lossy_data <- data %>% filter(sweep_type == "LossyQuery")

# ============================================================================
# Plot 1: Combined LSH vs LossyQuery (F1 vs p)
# ============================================================================

# For LSH: extract num_tables to group by
if (nrow(lsh_data) > 0) {
  lsh_data_plot <- lsh_data %>%
    extract(strategy,
            into = c("num_tables", "num_projections"),
            regex = "LSH-L([0-9]+)-K([0-9]+)",
            convert = TRUE,
            remove = FALSE) %>%
    mutate(strategy_group = paste0("LSH-L", num_tables))
} else {
  lsh_data_plot <- lsh_data
}

# Combine data with group column for plotting
plot_data <- bind_rows(
  lossy_data %>% mutate(strategy_group = as.character(strategy)),
  if (nrow(lsh_data) > 0) lsh_data_plot %>% select(-num_tables, -num_projections) else lsh_data
)

ggplot(plot_data, aes(x = p, y = f1, color = strategy_group, group = strategy_group)) +
  scale_x_reverse() +
  # All strategies with lines
  geom_line(alpha = 0.7, linewidth = 1) +
  geom_point(aes(shape = sweep_type), size = 2.5, alpha = 0.8) +
  scale_shape_manual(values = c("LSH" = 18, "LossyQuery" = 16)) +
  labs(
    title = sprintf("LSH vs LossyQuery Performance (n=%d, dim=%d)",
                    plot_data$num_nodes[1], plot_data$embedding_dim[1]),
    subtitle = "LSH grouped by num_tables (L), lines show varying projections (K)",
    x = "p (Recall of repelling edges)",
    y = "F1 Score",
    color = "Strategy/Config",
    shape = "Type"
  ) +
  theme_bw() +
  theme(legend.position = "right")

ggsave("f1_score_vs_p_combined.png", width = 12, height = 8)

# ============================================================================
# Plot 2: LSH Heatmap (F1 score)
# ============================================================================

if (nrow(lsh_data) > 0) {
  # Extract num_tables and num_projections from strategy
  lsh_data <- lsh_data %>%
    extract(strategy,
            into = c("num_tables", "num_projections"),
            regex = "LSH-L([0-9]+)-K([0-9]+)",
            convert = TRUE,
            remove = FALSE)

  ggplot(lsh_data, aes(x = num_projections, y = num_tables, fill = f1)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.3f", f1)), color = "black", size = 3) +
    scale_fill_gradient2(
      low = "red",
      mid = "yellow",
      high = "green",
      midpoint = median(lsh_data$f1),
      name = "F1 Score"
    ) +
    scale_x_continuous(breaks = unique(lsh_data$num_projections)) +
    scale_y_continuous(breaks = unique(lsh_data$num_tables)) +
    labs(
      title = sprintf("LSH F1 Score Heatmap (n=%d, dim=%d)",
                      lsh_data$num_nodes[1], lsh_data$embedding_dim[1]),
      x = "Number of Projections (K)",
      y = "Number of Tables (L)"
    ) +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

  ggsave("lsh_heatmap_f1.png", width = 10, height = 8)

  # ============================================================================
  # Plot 3: LSH Heatmap (Recall p)
  # ============================================================================

  ggplot(lsh_data, aes(x = num_projections, y = num_tables, fill = p)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.3f", p)), color = "black", size = 3) +
    scale_fill_gradient2(
      low = "red",
      mid = "yellow",
      high = "green",
      midpoint = median(lsh_data$p),
      name = "Recall (p)"
    ) +
    scale_x_continuous(breaks = unique(lsh_data$num_projections)) +
    scale_y_continuous(breaks = unique(lsh_data$num_tables)) +
    labs(
      title = sprintf("LSH Recall (p) Heatmap (n=%d, dim=%d)",
                      lsh_data$num_nodes[1], lsh_data$embedding_dim[1]),
      subtitle = "Average recall of repelling edges found by LSH vs ATree",
      x = "Number of Projections (K)",
      y = "Number of Tables (L)"
    ) +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

  ggsave("lsh_heatmap_recall.png", width = 10, height = 8)

  # ============================================================================
  # Plot 4: LSH Heatmap (Time)
  # ============================================================================

  ggplot(lsh_data, aes(x = num_projections, y = num_tables, fill = time_ms)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.0f", time_ms)), color = "black", size = 3) +
    scale_fill_gradient2(
      low = "green",
      mid = "yellow",
      high = "red",
      midpoint = median(lsh_data$time_ms),
      name = "Time (ms)"
    ) +
    scale_x_continuous(breaks = unique(lsh_data$num_projections)) +
    scale_y_continuous(breaks = unique(lsh_data$num_tables)) +
    labs(
      title = sprintf("LSH Runtime Heatmap (n=%d, dim=%d)",
                      lsh_data$num_nodes[1], lsh_data$embedding_dim[1]),
      subtitle = sprintf("%d iterations", unique(lsh_data$num_nodes)[1] * 2),
      x = "Number of Projections (K)",
      y = "Number of Tables (L)"
    ) +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

  ggsave("lsh_heatmap_time.png", width = 10, height = 8)

  # ============================================================================
  # Plot 5: LSH Heatmap (Cost: Failure Rate × Time)
  # ============================================================================

  # Cost metric: failure rate × time (lower is better)
  # Captures both quality and speed in a single metric
  # F1=0.9, 1000ms -> cost=100, F1=0.95, 1000ms -> cost=50 (twice as good!)
  lsh_data <- lsh_data %>%
    mutate(cost = (1 - f1) * time_ms)

  ggplot(lsh_data, aes(x = num_projections, y = num_tables, fill = cost)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.1f", cost)), color = "black", size = 3) +
    scale_fill_gradient2(
      low = "green",
      mid = "yellow",
      high = "red",
      midpoint = median(lsh_data$cost),
      name = "(1-F1)×ms"
    ) +
    scale_x_continuous(breaks = unique(lsh_data$num_projections)) +
    scale_y_continuous(breaks = unique(lsh_data$num_tables)) +
    labs(
      title = sprintf("LSH Cost: Failure Rate × Time (n=%d, dim=%d)",
                      lsh_data$num_nodes[1], lsh_data$embedding_dim[1]),
      subtitle = "Lower is better - captures both quality and speed",
      x = "Number of Projections (K)",
      y = "Number of Tables (L)"
    ) +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

  ggsave("lsh_heatmap_cost.png", width = 10, height = 8)

  # ============================================================================
  # Plot 6: LSH Heatmap (F1 per second - linear efficiency)
  # ============================================================================

  lsh_data <- lsh_data %>%
    mutate(f1_per_second = f1 / (time_ms / 1000))

  ggplot(lsh_data, aes(x = num_projections, y = num_tables, fill = f1_per_second)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = sprintf("%.3f", f1_per_second)), color = "black", size = 3) +
    scale_fill_gradient2(
      low = "red",
      mid = "yellow",
      high = "green",
      midpoint = median(lsh_data$f1_per_second),
      name = "F1/s"
    ) +
    scale_x_continuous(breaks = unique(lsh_data$num_projections)) +
    scale_y_continuous(breaks = unique(lsh_data$num_tables)) +
    labs(
      title = sprintf("LSH Efficiency: F1 Score per Second (n=%d, dim=%d)",
                      lsh_data$num_nodes[1], lsh_data$embedding_dim[1]),
      subtitle = "Higher is better - more quality per unit time",
      x = "Number of Projections (K)",
      y = "Number of Tables (L)"
    ) +
    theme_bw() +
    theme(panel.grid.major = element_blank(),
          panel.grid.minor = element_blank())

  ggsave("lsh_heatmap_efficiency.png", width = 10, height = 8)

}

# ============================================================================
# Plot 7: LossyQuery strategies
# ============================================================================

if (nrow(lossy_data) > 0) {
  ggplot(lossy_data, aes(x = p, y = f1, color = strategy)) +
    scale_x_reverse() +
    geom_point(size = 2) +
    geom_line(alpha = 0.7) +
    labs(
      title = sprintf("LossyQuery Strategy Sweep (n=%d, dim=%d)",
                      lossy_data$num_nodes[1], lossy_data$embedding_dim[1]),
      x = "p (Target recall parameter)",
      y = "F1 Score",
      color = "Strategy"
    ) +
    theme_bw()

  ggsave("lossy_strategies.png", width = 10, height = 8)

  # ============================================================================
  # Plot 8: LossyQuery runtime comparison
  # ============================================================================

  ggplot(lossy_data, aes(x = p, y = time_ms, color = strategy)) +
    scale_x_reverse() +
    geom_point(size = 2) +
    geom_line(alpha = 0.7) +
    labs(
      title = sprintf("LossyQuery Runtime vs p (n=%d, dim=%d)",
                      lossy_data$num_nodes[1], lossy_data$embedding_dim[1]),
      x = "p (Target recall parameter)",
      y = "Runtime (ms)",
      color = "Strategy"
    ) +
    theme_bw()

  ggsave("lossy_runtime.png", width = 10, height = 8)
}

# ============================================================================
# Summary Statistics
# ============================================================================

cat("\n=== Sweep Summary ===\n")
cat(sprintf("Graph: %d nodes, dimension %d\n",
            data$num_nodes[1], data$embedding_dim[1]))

if (nrow(lsh_data) > 0) {
  cat("\nLSH Results:\n")
  best_f1_idx <- which.max(lsh_data$f1)
  cat(sprintf("  Best F1: %.4f at L=%d, K=%d (p=%.4f, time=%.0fms)\n",
              lsh_data$f1[best_f1_idx],
              lsh_data$num_tables[best_f1_idx],
              lsh_data$num_projections[best_f1_idx],
              lsh_data$p[best_f1_idx],
              lsh_data$time_ms[best_f1_idx]))

  best_p_idx <- which.max(lsh_data$p)
  cat(sprintf("  Best recall: %.4f at L=%d, K=%d (f1=%.4f, time=%.0fms)\n",
              lsh_data$p[best_p_idx],
              lsh_data$num_tables[best_p_idx],
              lsh_data$num_projections[best_p_idx],
              lsh_data$f1[best_p_idx],
              lsh_data$time_ms[best_p_idx]))

  fastest_idx <- which.min(lsh_data$time_ms)
  cat(sprintf("  Fastest: %.0fms at L=%d, K=%d (f1=%.4f, p=%.4f)\n",
              lsh_data$time_ms[fastest_idx],
              lsh_data$num_tables[fastest_idx],
              lsh_data$num_projections[fastest_idx],
              lsh_data$f1[fastest_idx],
              lsh_data$p[fastest_idx]))

  best_efficiency_idx <- which.max(lsh_data$f1_per_second)
  cat(sprintf("  Best efficiency: %.3f F1/s at L=%d, K=%d (f1=%.4f, time=%.0fms)\n",
              lsh_data$f1_per_second[best_efficiency_idx],
              lsh_data$num_tables[best_efficiency_idx],
              lsh_data$num_projections[best_efficiency_idx],
              lsh_data$f1[best_efficiency_idx],
              lsh_data$time_ms[best_efficiency_idx]))

  lowest_cost_idx <- which.min(lsh_data$cost)
  cat(sprintf("  Lowest cost: %.1f (1-F1)×ms at L=%d, K=%d (f1=%.4f, time=%.0fms)\n",
              lsh_data$cost[lowest_cost_idx],
              lsh_data$num_tables[lowest_cost_idx],
              lsh_data$num_projections[lowest_cost_idx],
              lsh_data$f1[lowest_cost_idx],
              lsh_data$time_ms[lowest_cost_idx]))
}

if (nrow(lossy_data) > 0) {
  cat("\nLossyQuery Results:\n")
  best_f1_idx <- which.max(lossy_data$f1)
  cat(sprintf("  Best F1: %.4f with %s at p=%.1f (time=%.0fms)\n",
              lossy_data$f1[best_f1_idx],
              lossy_data$strategy[best_f1_idx],
              lossy_data$p[best_f1_idx],
              lossy_data$time_ms[best_f1_idx]))
}

cat("\nGenerated plots:\n")
cat("  - f1_score_vs_p_combined.png (LSH vs LossyQuery)\n")
if (nrow(lsh_data) > 0) {
  cat("  - lsh_heatmap_f1.png (LSH F1 heatmap)\n")
  cat("  - lsh_heatmap_recall.png (LSH recall heatmap)\n")
  cat("  - lsh_heatmap_time.png (LSH runtime heatmap)\n")
  cat("  - lsh_heatmap_cost.png (LSH cost: failure rate × time)\n")
  cat("  - lsh_heatmap_efficiency.png (LSH F1/s efficiency heatmap)\n")
}
if (nrow(lossy_data) > 0) {
  cat("  - lossy_strategies.png (LossyQuery F1 comparison)\n")
  cat("  - lossy_runtime.png (LossyQuery runtime comparison)\n")
}
