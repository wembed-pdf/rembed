# Shared theme configuration for all plots
# Source this file at the top of each R script:
#   source("../theme.R")  # adjust path as needed

suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(showtext))

# If Path R_SHOWTEXT_FONTPATH is set use that, assuming it points to the lmodern fonts from Nix.
if (!is.null(Sys.getenv("R_SHOWTEXT_FONTPATH"))) {
  # message("Using lmodern fonts from R_SHOWTEXT_FONTPATH:", Sys.getenv("R_SHOWTEXT_FONTPATH"))
  font_add(
    family = "lmodern",
    regular = paste0(Sys.getenv("R_SHOWTEXT_FONTPATH"), "/lmroman10-regular.otf"),
    bold = paste0(Sys.getenv("R_SHOWTEXT_FONTPATH"), "/lmroman10-bold.otf"),
    italic = paste0(Sys.getenv("R_SHOWTEXT_FONTPATH"), "/lmroman10-italic.otf")
  )
} else {
  message("R_SHOWTEXT_FONTPATH not set, using default fonts. To use lmodern fonts, set R_SHOWTEXT_FONTPATH to the directory containing lmodern .otf files.")
}

showtext_auto()

# =============================================================================
# Plot Dimensions
# =============================================================================
# For LaTeX with \includegraphics[width=0.45\textwidth], assuming ~6.5in text width
# Single plot: 0.45 * 6.5 = ~2.9in displayed, but we render larger for quality

PLOT_WIDTH_SINGLE <- 2.5 # inches (for 0.45\textwidth figures)
PLOT_WIDTH_FULL <- 5 # inches (for full-width figures)
PLOT_HEIGHT <- 2.5 # inches (maintains good aspect ratio)
PLOT_DPI <- 300 # publication quality
PLOT_RATIO <- 0.7 # Height/Width ratio 

# =============================================================================
# Font Sizes
# =============================================================================
BASE_SIZE <- 10
TITLE_SIZE <- 10
AXIS_TITLE_SIZE <- 8
AXIS_TEXT_SIZE <- 7
LEGEND_TEXT_SIZE <- 7
LEGEND_TITLE_SIZE <- 8
STRIP_TEXT_SIZE <- 8

# =============================================================================
# Colors
# =============================================================================
# Consistent color palette for algorithms

ALGO_COLORS <- c(
  "atree" = "#E41A1C",
  "naive_atree" = "#2e1ae4",
  "atree2" = "#2e1ae4",
  "wembed_snn" = "#FF7F00",
  "brute-force" = "#984EA3",
  "kiddo" = "#4DAF4A",
  "agrid" = "#377EB8",
  "neighbourhood" = "#F781BF",
  "boost_rtree" = "#A65628",
  # not used as often
  "vptree" = "#c6a1a1",
  "sklearn_balltree" = "#000000",
  "sklearn_kdtree" = "#FB8072",
  "kd_tree" = "#BEBADA",
  "sif" = "#D9D9D9",
  "nabo" = "#FFB347",
  "nanoflann" = "#CFCFC4",
  "cgal_kdtree" = "#8e8e1e",
  "ball_tree" = "#b3ffda",
  "SNN" = "#FF7F00",
  # "py_snn" = "#FF7F00",
  "py_snn" = "#e57b11",
  "line-lsh" = "#FDB462",
  "lsh-2" = "#B3DE69",
  "quadtree" = "#cfd91a",
  "grid" = "#064311",
  "dyn_atree" = "#9e0142",
  "orthtree" = "#8071c3",
  "snn" = "#FF7F00",
  "naive_snn" = "#6f3c09",
  "naive_atree_non_progressive" = "#2e1ae4"
)

KITColors <- c(
  "KITgreen" = "#009682",
  "KITgreen70" = "#3cbab1",
  "KITgreen50" = "#7fd1c0",
  "KITgreen30" = "#b3e7d0",
  "KITgreen15" = "#e6f3e0",
  "KITwhite" = "#ffffff",
  "KITwhite70" = "#ffffff",
  "KITwhite50" = "#ffffff",
  "KITwhite30" = "#ffffff",
  "KITwhite15" = "#ffffff",
  "KITdarkblue" = "#002d4c",
  "KITdarkblue70" = "#3b5c7e",
  "KITdarkblue50" = "#7f8db0",
  "KITdarkblue30" = "#c2bfe0",
  "KITdarkblue15" = "#e6d9f0",
  "KITiceblue" = "#a8b9c4",
  "KITiceblue70" = "#c2d1d9",
  "KITiceblue50" = "#d9e1ec",
  "KITiceblue30" = "#e6ebf0",
  "KITiceblue15" = "#f2f5f8",
  "KITicegray" = "#dae1e6",
  "KITicegray70" = "#e6ebf0",
  "KITicegray50" = "#f2f5f8",
  "KITicegray30" = "#f9fbfc",
  "KITicegray15" = "#fcfdfe",
  "KITpinegreen" = "#005a50",
  "KITpinegreen70" = "#3b7f74",
  "KITpinegreen50" = "#7f9f9e",
  "KITpinegreen30" = "#c2d0c9",
  "KITpinegreen15" = "#e6e8e4",
  "KITcyan" = "#23a1e0",
  "KITcyan70" = "#5ab7e8",
  "KITcyan50" = "#8fd0ed",
  "KITcyan30" = "#c2e0f2",
  "KITcyan15" = "#e6f0f5",
  "KITblue" = "#0c537e",
  "KITblue70" = "#3b7f9e",
  "KITblue50" = "#7f9fbe", 
  "KITblue30" = "#c2d0e9",
  "KITblue15" = "#e6e8f4",
  "KITlightgreen" = "#8cb63c",
  "KITlightgreen70" = "#b3d16e",
  "KITlightgreen50" = "#d9e8a0",
  "KITlightgreen30" = "#e6f0c2",
  "KITlightgreen15" = "#f2f5e1",
  "KITforestgreen" = "#276738",
  "KITforestgreen70" = "#5ab77e",
  "KITforestgreen50" = "#8fd0a5",
  "KITforestgreen30" = "#c2e0cb",
  "KITforestgreen15" = "#e6f0e5",
  "KITblack" = "#000000",
  "KITblack70" = "#4d4d4d",
  "KITblack50" = "#7f7f7f",
  "KITblack30" = "#b3b3b3",
  "KITblack15" = "#e6e6e6",
  "KITpurple" = "#a3107c",
  "KITpurple70" = "#d16eb9",
  "KITpurple50" = "#e8b5d6",
  "KITpurple30" = "#f0d9e5",
  "KITpurple15" = "#f7ebf2",
  "KITorange" = "#df9b1b",
  "KITorange70" = "#f2c27b",
  "KITorange50" = "#f7e0c2",
  "KITorange30" = "#f9f0e1",
  "KITorange15" = "#fcf8f0",
  "KITyellow" = "#fce500",
  "KITyellow70" = "#fef2a6",
  "KITyellow50" = "#fef9d1",
  "KITyellow30" = "#fefcf0",
  "KITyellow15" = "#fefef8",
  "KITred" = "#d30015",
  "KITred70" = "#f2a1a1",
  "KITred50" = "#f7c2c2",
  "KITred30" = "#f9e0e0",
  "KITred15" = "#fcf0f0",
  "KITbrown" = "#a7822e",
  "KITbrown70" = "#d1b36e",
  "KITbrown50" = "#e8d1a5",
  "KITbrown30" = "#f0e8c2",
  "KITbrown15" = "#f7f0e1",
  "KITUndefined" = "#B7FFF5",
  "KITUndefined70" = "#d9fff8",
  "KITUndefined50" = "#e8fffb",
  "KITUndefined30" = "#f0fff8",
  "KITUndefined15" = "#f7fffc"
)

ALGO_KIT_COLORS <- c(
  "atree" = KITColors[["KITred"]],
  "naive_atree" = KITColors[["KITblue"]],
  "atree2" = KITColors[["KITblue"]],
  "wembed_snn" = KITColors[["KITorange"]],
  "brute-force" = KITColors[["KITcyan70"]],
  "kiddo" = "#4DAF4A",
  "agrid" = KITColors[["KITcyan"]],
  "neighbourhood" = KITColors[["KITpurple70"]],
  "boost_rtree" = KITColors[["KITiceblue70"]],
  # not used as often
  "vptree" = KITColors[["KITbrown50"]],
  "sklearn_balltree" = KITColors[["KITlightgreen70"]],
  "sklearn_kdtree" = KITColors[["KITyellow"]],
  "kd_tree" = KITColors[["KITiceblue"]],
  "sif" = KITColors[["KITUndefined"]],
  "nabo" = KITColors[["KITorange"]],
  "nanoflann" = KITColors[["KITbrown"]],
  "cgal_kdtree" = KITColors[["KITblack70"]],
  "ball_tree" = KITColors[["KITcyan"]],
  "SNN" = KITColors[["KITorange"]],
  # "py_snn" = KITColors["KITorange"],
  "py_snn" = "#e57b11",
  "line-lsh" = KITColors[["KITyellow"]],
  "lsh-2" = KITColors[["KITlightgreen"]],
  "quadtree" = KITColors[["KITiceblue"]],
  "grid" = KITColors[["KITdarkblue50"]],
  "dyn_atree" = "#9e0142",
  "orthtree" = "#8071c3",
  "snn" = KITColors[["KITorange"]],
  "naive_snn" = "#6f3c09",
  "naive_atree_non_progressive" = "#2e1ae4"
)

ALGO_COLORS <- ALGO_KIT_COLORS
# ALGO_COLORS <- c()
# Display Names
DISPLAY_NAMES <- c(
  "atree" = "SPRK-tree",
  "naive_atree" = "Naive SPRK-tree",
  "atree2" = "Atree2",
  "wembed_snn" = "wembed-SNN",
  "brute-force" = "Brute-Force",
  "kiddo" = "Kiddo",
  "agrid" = "Agrid",
  "neighbourhood" = "Neighbourhood",
  "vptree" = "VP-tree",
  "boost_rtree" = "Boost R-tree",
  "sklearn_balltree" = "sklearn Balltree",
  "sklearn_kdtree" = "sklearn KD-tree",
  "kd_tree" = "KD-tree",
  "sif" = "SIF",
  "nabo" = "nabo",
  "nanoflann" = "nanoflann",
  "cgal_kdtree" = "CGAL KD-tree",
  "ball_tree" = "Balltree",
  "SNN" = "SNN",
  "py_snn" = "SNN",
  "line-lsh" = "Line LSH",
  "lsh-2" = "LSH-2",
  "quadtree" = "Quadtree",
  "grid" = "Uniform Grid",
  "dyn_atree" = "Dyn Atree",
  "orthtree" = "Orthtree",
  "snn" = "SNN",
  "naive_snn" = "Naive SNN",
  "naive_atree_non_progressive" = "Naive Pruning SPRK-tree"
)

#Shapes for algorithms (if needed)
ALGO_SHAPES <- c(
  "atree" = 16, # solid circle
  "naive_atree" = 1, # circle with border
  "atree2" = 17, # solid triangle
  "wembed_snn" = 15, # solid square
  "brute-force" = 22, # square with white fill
  "kiddo" = 17, # solid triangle
  "agrid" = 20, # small solid circle
  "neighbourhood" = 21, # circle with white fill
  "boost_rtree" = 8, # asterisk
  # not used as often
  "vptree" = 23, # diamond with white fill
  "sklearn_balltree" = 4, # cross
  "sklearn_kdtree" = 8, # asterisk
  "kd_tree" = 4, # cross
  "sif" = 4, # cross
  "nabo" = 5, # diamond with border
  "nanoflann" = 6, # inverted triangle with border
  "cgal_kdtree" = 7, # square with border
  "ball_tree" = 9, # circle with border
  "SNN" = 15, # solid square
  "py_snn" = 15, # solid square
  "line-lsh" = 17, # solid triangle
  "lsh-2" = 18, # solid diamond
  "quadtree" = 19, # solid circle with border
  "grid" = 18, # solid diamond
  "dyn_atree" = 16, # solid circle
  "orthtree" = 20, # small solid circle
  "snn" = 15, # solid square
  "naive_snn" = 15, # solid square
  "naive_atree_non_progressive" = 15 # solid square
)



ALGO_LEGEND_ORDER <- c(
  "sklearn_kdtree",
  "cgal_kdtree",
  "nanoflann",
  "nabo",
  "neighbourhood",
  "kiddo",
  "sklearn_balltree",
  "py_snn",
  "vptree",
  "boost_rtree",
  "brute-force",
  "orthtree",
  "grid",
  "atree",
  # currently unused structures
  "naive_snn",
  "snn",
  "wembed_snn",
  "SNN",
  "naive_atree",
  "atree2",
  "agrid",
  "kd_tree",
  "sif",
  "ball_tree",
  "line-lsh",
  "lsh-2",
  "quadtree",
  "dyn_atree",
  "naive_atree_non_progressive"
)

# =============================================================================
# Custom Theme
# =============================================================================
theme_paper <- function(base_size = BASE_SIZE) {
  theme_bw(base_size = base_size) %+replace%
    theme(
      text = element_text(family = "lmodern"),
      # Title
      plot.title = element_text(
        size = TITLE_SIZE,
        face = "bold",
        hjust = 0.5,
        margin = margin(b = 10)
      ),
      plot.subtitle = element_text(
        size = AXIS_TITLE_SIZE,
        hjust = 0.5,
        margin = margin(b = 10)
      ),

      # Axis titles
      axis.title = element_text(size = AXIS_TITLE_SIZE),
      # axis.title.x = element_text(margin = margin(t = 8)),
      # axis.title.y = element_text(angle = 90, margin = margin(r = 8)),
      axis.title.x = element_text(margin = margin(t = 2)),
      axis.title.y = element_text(angle = 90, margin = margin(r = 2)),

      # Axis text
      axis.text = element_text(size = AXIS_TEXT_SIZE, color = "black"),

      # Legend
      legend.title = element_text(size = LEGEND_TITLE_SIZE),
      legend.text = element_text(size = LEGEND_TEXT_SIZE),
      legend.position = "bottom",
      legend.background = element_rect(fill = "white", color = NA),
      legend.key = element_rect(fill = "white", color = NA),
      legend.margin = margin(t = -5),

      # Panel
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA, linewidth = 0.5),

      # Strip (for faceted plots)
      # strip.background = element_rect(fill = "grey90", color = "black"),
      strip.background = element_blank(),
      strip.text = element_text(size = STRIP_TEXT_SIZE, face = "plain", margin = margin(b = 5, t = 1)),

      # Margins
      plot.margin = margin(t = 5, 0, 0, 0) # prevent clipping of top plot border when no title is present
    )
}

# =============================================================================
# Helper Functions
# =============================================================================

# Save plot with consistent settings
save_plot <- function(filename, plot = last_plot(),
                      width = PLOT_WIDTH_SINGLE,
                      height = PLOT_HEIGHT,
                      dpi = PLOT_DPI) {
  ggsave(
    filename = filename,
    plot = plot,
    width = width,
    height = height,
    dpi = dpi,
    bg = "white"
  )
  message(paste("Saved:", filename, "(", width, "x", height, "in @", dpi, "dpi)"))
}

# Save full-width plot
save_plot_full <- function(filename, plot = last_plot(),
                           height = PLOT_HEIGHT,
                           dpi = PLOT_DPI) {
  save_plot(filename, plot, width = PLOT_WIDTH_FULL, height = height, dpi = dpi)
}

# Consistent scale for algorithms
scale_color_algo <- function(...) {
  scale_color_manual(values = ALGO_COLORS, labels = DISPLAY_NAMES, breaks = ALGO_LEGEND_ORDER, ...)
}

# Consistent scale for algorithms
scale_shape_algo <- function(...) {
  scale_shape_manual(values = ALGO_SHAPES,labels = DISPLAY_NAMES, breaks = ALGO_LEGEND_ORDER, ...)
}

# Consistent fill scale for algorithms (if needed)
scale_fill_algo <- function(...) {
  scale_fill_manual(values = ALGO_COLORS, labels = DISPLAY_NAMES, breaks = ALGO_LEGEND_ORDER, ...)
}

# Log scale with better formatting
scale_x_log10_nice <- function(...) {
  scale_x_log10(
    labels = scales::label_number(scale_cut = scales::cut_short_scale()),
    ...
  )
}

scale_y_log10_nice <- function(...) {
  scale_y_log10(
    labels = scales::label_number(scale_cut = scales::cut_short_scale()),
    ...
  )
}

scale_y_log10_nanoseconds <- function(...) {
  scale_y_log10(
    labels = scales::label_number(scale_cut = scales::cut_si("s")),
    ...
  )
}
# =============================================================================
# Set as default theme
# =============================================================================
theme_set(theme_paper())
