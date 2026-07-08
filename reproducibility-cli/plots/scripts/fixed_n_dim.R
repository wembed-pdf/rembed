source("plots/scripts/theme.R")

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(dotenv))
suppressPackageStartupMessages(library(scales))

df <- read.csv("benchmark_results/embedding_benchmark_results.csv", stringsAsFactors = FALSE)
output_folder <- "plots/"

structures <- c(
    "vptree",
    "grid",
    "orthtree",
    "sklearn_balltree",
    "boost_rtree",
    "py_snn",
    "atree",
    "ball_tree",
    "brute-force",
    # "kd_tree",
    # "line-lsh",
    # "lsh-2",
    "neighbourhood",
    "kiddo"
    # "sif"
    # "SNN",
    # "nabo",
    # "sklearn_kdtree",
    # "wembed_snn",
    # "nanoflann",
    # "agrid",
    # "cgal_kdtree",
    # "snn"
    # "naive_snn",
    # "naive_atree"
)

plot_distributions <- function(df, filename = "benchmark_fixed_n.pdf", measurement_x, measurement_y, facet_wrap_var) {
    p <- ggplot(df, aes(x = .data[[measurement_x]], y = .data[[measurement_y]], color = data_structure)) +
        geom_line(aes(color = data_structure), linewidth = 0.5) +
        geom_point(aes(shape = data_structure), size = 1.25) +
        scale_color_algo() +
        scale_shape_algo() +
        scale_y_log10(
            labels = function(x) {
                ifelse(x / 1000 >= 1000,
                    paste0(x / 1000 / 1000, "ms"),
                    paste0(x / 1000, "µs")
                )
            }
        ) +
        labs(
            x = "Embedding Dimension",
            y = ""
        ) +
        theme_paper() +
        theme(
            plot.title = element_text(hjust = 0.5),
            legend.title = element_blank(),
            legend.position = "bottom",
            legend.margin = margin(t = 1, r = 0, b = 0, l = 0),
            legend.key.height = unit(0.2, "line"),
            aspect.ratio = PLOT_RATIO,
            plot.margin = margin(l = -10),
        ) 

    if (facet_wrap_var == "node_count") {
        p <- p + facet_wrap(~node_count, nrow = 1, labeller = labeller(node_count = function(x) paste("Vertices:", format(as.numeric(as.character(x)), big.mark = " ", scientific = FALSE, trim = TRUE))), scales = "free_y")
    } else if (facet_wrap_var == "dimension") {
        p <- p + scale_x_log10_nice()
        p <- p + facet_wrap(~dimension, nrow = 1, labeller = labeller(dimension = function(x) paste("Embedding Dimension:", x)), scales = "free_y")
    }
    ggsave(filename = paste0(output_folder, filename), plot = p, width = 5.5, height = 1.9, dpi = 300, device = cairo_pdf)
    message(paste0("Plot saved to ", output_folder, filename))
}

measurement <- "wall_time_mean_ns"

# Filter some orthtree results that are too large for better visualization
df <- df %>%
    filter(!(data_structure == "orthtree" & dimension >= 14 & node_count == 10000)) %>%
    filter(!(data_structure == "orthtree" & dimension >= 10 & node_count == 100000))

stddev_measurement <- sub("_mean", "_stddev", measurement)

# double times for brute-force
# since brute-force queries only from lower to higher node_ids
df <- df %>%
    mutate(
        !!sym(measurement) := ifelse(data_structure == "brute-force", !!sym(measurement) * 2, !!sym(measurement)),
        !!sym(stddev_measurement) := ifelse(data_structure == "brute-force", !!sym(stddev_measurement) * 2, !!sym(stddev_measurement))
    )

df <- df %>%
    filter(data_structure %in% structures) %>%
    filter(!(data_structure == "orthtree" & dimension == 16))

# average instructions_mean for each data_structure and dimension
df <- df %>%
    group_by(data_structure, dimension, node_count) %>%
    summarise(
        instructions_mean = mean(instructions_mean, na.rm = TRUE),
        cycles_mean = mean(cycles_mean, na.rm = TRUE),
        wall_time_mean_ns = mean(wall_time_mean_ns, na.rm = TRUE),
        instructions_stddev = mean(instructions_stddev, na.rm = TRUE),
        cycles_stddev = mean(cycles_stddev, na.rm = TRUE),
        wall_time_stddev_ns = mean(wall_time_stddev_ns, na.rm = TRUE),
        .groups = "drop"
    )

#change order of algorithms for better drawing (first drawn will be in the back, last drawn will be in the front)
df$data_structure <- factor(df$data_structure, levels = structures)
df <- df %>%
    arrange(data_structure, node_count, dimension) %>%
    filter(!(data_structure == "grid" & dimension == 16))

df_main_bench <- df %>%
    filter(node_count %in% c(10000, 100000, 1000000)) %>%
    filter(dimension <= 16)

plot_distributions(df = df_main_bench, filename = "figure_3.pdf", measurement_x = "dimension", measurement_y = measurement, facet_wrap_var = "node_count")

df_half_log_appendix <- df %>%
    filter(dimension %in% c(2, 8, 16))

plot_distributions(df = df_half_log_appendix, filename = "figure_9.pdf", measurement_x = "node_count", measurement_y = measurement, facet_wrap_var = "dimension")