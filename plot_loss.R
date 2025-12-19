#!/usr/bin/env Rscript


suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))

measurements <- read.csv("loss_measurements_4.csv")

# Expect columns: strategy, p, f1, precision, recall

# Group by strategy

# Plot F1 Score vs p, connect points of the same strategy with lines
ggplot(measurements, aes(x = p, y = f1, color = strategy)) +
  # reverse the x-axis
  scale_x_reverse() +
  geom_point() +
  geom_line() +
  labs(
    title = "Variants of Approximation Losses dim=4; n=10000",
    x = "p (Recall of repelling edges)",
    y = "F1 Score"
  ) +
  theme_bw()


ggsave("f1_score_vs_p_4.png")
