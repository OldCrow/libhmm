#!/usr/bin/env Rscript
#
# prepare_elk_data.R — Export elk GPS step lengths and turning angles for elk_movement_example
#
# Extracts per-animal step lengths and turning angles from the moveHMM R package
# elk_data dataset and writes one CSV per animal to ELK_DATA_DIR (default: /tmp).
#
# Usage:
#   Rscript scripts/prepare_elk_data.R [output_dir]
#
# Output files:
#   elk_115_obs.csv, elk_163_obs.csv, elk_287_obs.csv, elk_363_obs.csv
#   Each file has two columns: step, angle
#
# Then build and run the C++ example:
#   cmake --build build --config Release --target elk_movement_example
#   ./build/examples/Release/elk_movement_example [output_dir]
#
# Requirements:
#   install.packages("moveHMM")
#
# Data source:
#   Morales et al. (2004). Extracting more out of relocation data: building
#   movement models as mixtures of random walks. Ecology, 85(9), 2436-2445.
#   Bundled in: Michelot T, Langrock R, Patterson TA (2016). moveHMM.
#   Methods in Ecology and Evolution, 7(11), 1308-1315.
#

args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) > 0) args[1] else "/tmp"

if (!require("moveHMM", quietly = TRUE)) {
    stop("moveHMM not installed. Run: install.packages('moveHMM')")
}

data(elk_data)
elk <- prepData(elk_data, type = "UTM", coordNames = c("Easting", "Northing"))

ids <- unique(elk$ID)
cat("Exporting step lengths to", out_dir, "\n")

for (id in ids) {
    sub <- elk[elk$ID == id & !is.na(elk$step) & elk$step > 0 & !is.na(elk$angle), ]
    safe_id <- gsub("-", "_", id)
    fname <- file.path(out_dir, paste0(safe_id, "_obs.csv"))
    write.csv(data.frame(step = sub$step, angle = sub$angle), fname, row.names = FALSE)
    cat("  ", id, "->", fname, "(", nrow(sub), "obs )\n")
}

cat("\nDone. Run elk_movement_example", out_dir, "\n")
