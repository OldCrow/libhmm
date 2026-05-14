#!/usr/bin/env Rscript
#
# prepare_wind_data.R — Download NOAA ISD wind data for wind_direction_example
#
# Downloads hourly wind observations from Chicago O'Hare International
# Airport (NOAA ISD station 725300-14819) for 2015, extracts wind direction
# and speed, and exports to a CSV file.
#
# Usage:
#   Rscript scripts/prepare_wind_data.R [output_dir]
#
# Output:
#   ohare_wind_2015.csv — direction_rad, speed_ms columns
#
# Then build and run:
#   cmake --build build --config Release --target wind_direction_example
#   ./build/examples/Release/wind_direction_example [output_dir]
#
# Data source:
#   NOAA NCEI (2001): Global Surface Hourly [ISD]. NCEI.
#   https://www.ncei.noaa.gov/data/global-hourly/
#   Station: Chicago O'Hare (USAF 725300, WBAN 14819)
#

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) > 0) args[1] else "/tmp"

options(timeout = 120)

# Download O'Hare 2015 CSV from NCEI
url <- "https://www.ncei.noaa.gov/data/global-hourly/access/2015/72530094846.csv"
dest <- tempfile(fileext = ".csv")

cat("Downloading NOAA ISD data for Chicago O'Hare (2015)...\n")
tryCatch(
    download.file(url, dest, method = "curl", quiet = TRUE),
    error = function(e) stop("Download failed: ", e$message)
)

df <- read.csv(dest, stringsAsFactors = FALSE)
cat("Downloaded", nrow(df), "rows\n")

# Parse WND field: "direction,quality,type,speed,quality2"
# direction: degrees (999 = missing), speed: m/s * 10 (9999 = missing)
wnd      <- strsplit(df$WND, ",")
dir_deg  <- as.numeric(sapply(wnd, function(x) x[1]))
spd_raw  <- as.numeric(sapply(wnd, function(x) x[4]))

valid <- !is.na(dir_deg) & dir_deg >= 0 & dir_deg <= 360 &
         !is.na(spd_raw) & spd_raw < 9000 & spd_raw > 0

dir_rad  <- dir_deg[valid] * pi / 180
speed_ms <- spd_raw[valid] / 10.0

wind <- data.frame(direction_rad = dir_rad, speed_ms = speed_ms)
wind <- wind[complete.cases(wind), ]

cat("Valid wind observations:", nrow(wind), "\n")
cat("Speed range:", round(min(wind$speed_ms), 1), "to",
    round(max(wind$speed_ms), 1), "m/s\n")

out_path <- file.path(out_dir, "ohare_wind_2015.csv")
write.csv(wind, out_path, row.names = FALSE)
cat("\nExported:", out_path, "(", nrow(wind), "rows )\n")
cat("\nDone. Run: wind_direction_example", out_dir, "\n")
