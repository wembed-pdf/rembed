source("plots/scripts/theme.R")

suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(dotenv))
suppressPackageStartupMessages(library(scales))

output_folder <- "plots/"

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript script.R <argument>\n")
  quit(status = 1)
}

data_file <- args[1]

# Read the CSV file
# dimension,node_count,radius,distribution,structure,wall_time_ns,wall_time_std_ns,instructions,instructions_std,cycles,cycles_std,samples
message(paste("Reading data from", data_file))
data <- fread(data_file)
df <- data

# measurement <- "cycles_mean"
measurement <- "wall_time_mean_ns"
# measurement <- "instruction_count_mean"

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
    "kiddo",
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
    "naive_atree"
)

dim_2_structures <- c(
    "quadtree",
    # "grid",
    "quadtree_2",
    "grid_2"
)

df <- df %>%
    filter(!(data_structure%in% dim_2_structures) | dimension == 2) %>%
    filter(dimension %in% c(2, 8, 32)) %>%
    filter(node_count <= 10000000) %>%
    filter(data_structure%in% structures)

stddev_measurement <- sub("mean_ns", "stddev_ns", measurement)

# scale time to seconds
if (grepl("_ns", measurement)) {
    df <- df %>%
        mutate(
            !!sym(measurement) := !!sym(measurement) / 1e9,
            !!sym(stddev_measurement) := !!sym(stddev_measurement) / 1e9
        )
}

plot_distributions <- function(df, measurement, output_folder, filename = "figure_4.pdf") {
    suppressWarnings({
        p <- ggplot(df, aes(x = node_count, y = .data[[measurement]], color = data_structure)) +
            geom_line(linewidth = 0.5) +
            geom_point(aes(shape = data_structure), size = 1) +
            scale_x_log10_nice() +
            scale_y_log10_nanoseconds() +
            scale_color_algo() +
            scale_shape_algo() +
            labs(
                x = "Number of Points (log scale)",
                y = ""
            ) +
            theme_paper() +
            theme(
                plot.title = element_text(hjust = 0.5),
                legend.title = element_blank(),
                legend.position = "bottom",
            legend.margin = margin(t = 1, r = 0, b = 0, l = 0),
                plot.margin = margin(r = 5, t = 2, l = -10),
                            legend.key.height = unit(0.2, "line"),
            strip.text = element_text(size = 7, face = "bold", margin = margin(b = 5, t = 1)),
            aspect.ratio = PLOT_RATIO,
            ) +
            facet_wrap(
                ~dimension,
                nrow = 1,
                labeller = labeller(dimension = function(x) paste("Dimension:", x)),
                scales = "free_y"
            )

        out_path <- paste0(output_folder, filename)
        ggsave(filename = out_path, plot = p, dpi = 300, width = 5.5, height = 1.9, bg = "white")
        message("Plot saved to ", out_path)
    })
}

df_filtered <- df %>%
    filter(data_structure%notin% c(
        "orthtree",
        "py_snn",
        "boost_rtree",
        "grid",
        "py_snn",
        "neighbourhood"
    ))
df_no_naive <- df %>%
    filter(data_structure!= "naive_atree")
plot_distributions(df_no_naive, measurement, output_folder)
plot_distributions(df_filtered, measurement, output_folder, filename = "figure_11.pdf")