source("plots/scripts/theme.R")

# Get command line arguments
args <- commandArgs(trailingOnly = TRUE)

map_alias <- function(x) {
    if (!is.null(DISPLAY_NAMES[[x]])) {
        return(DISPLAY_NAMES[[x]])
    } else {
        return("INVALID DISPLAY NAME")
    }
}

output <- sapply(args, map_alias)

cat(output, sep = "\n")
