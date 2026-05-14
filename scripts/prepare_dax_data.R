#!/usr/bin/env Rscript
#
# prepare_dax_data.R — Download DAX daily log-returns for dax_regime_example
#
# Downloads ^GDAXI (DAX) closing prices from Yahoo Finance, 2000-01-01 to
# 2022-12-31, and exports daily log-returns to ELK_DATA_DIR (default: /tmp).
#
# Usage:
#   Rscript scripts/prepare_dax_data.R [output_dir]
#
# Output files:
#   dax_logreturns.csv   — one column: logreturn
#   dax_2000_2022.csv    — Date, Close, logreturn (for inspection)
#
# Then build and run the C++ example:
#   cmake --build build --config Release --target dax_regime_example
#   ./build/examples/Release/dax_regime_example [output_dir]
#
# Requirements:
#   install.packages("quantmod")
#
# Reference:
#   fHMM: Hidden Markov Models for Financial Time Series in R
#   Oelschläger, Adam & Michels (2024), J. Statistical Software, 109(9).
#   https://doi.org/10.18637/jss.v109.i09
#

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) > 0) args[1] else "/tmp"

options(timeout = 120)
if (!require("quantmod", quietly = TRUE)) {
    message("Installing quantmod...")
    install.packages("quantmod", repos = "https://cran.r-project.org",
                     quiet = TRUE, type = "binary")
    library(quantmod, quietly = TRUE)
}

cat("Downloading ^GDAXI from Yahoo Finance (2000-01-01 to 2022-12-31)...\n")
suppressWarnings(
    getSymbols("^GDAXI", from = "2000-01-01", to = "2022-12-31",
               auto.assign = TRUE, warnings = FALSE)
)

dax <- as.data.frame(GDAXI)
dax$Date  <- as.Date(rownames(dax))
dax       <- dax[, c("Date", "GDAXI.Close")]
names(dax)[2] <- "Close"
dax       <- dax[!is.na(dax$Close), ]
dax       <- dax[order(dax$Date), ]

dax$logreturn <- c(NA, diff(log(dax$Close)))
dax           <- dax[!is.na(dax$logreturn), ]

cat("Downloaded", nrow(dax), "daily log-returns\n")
cat("Date range:", format(min(dax$Date)), "to", format(max(dax$Date)), "\n")

write.csv(dax[, c("Date", "Close", "logreturn")],
          file.path(out_dir, "dax_2000_2022.csv"), row.names = FALSE)

write.csv(data.frame(logreturn = dax$logreturn),
          file.path(out_dir, "dax_logreturns.csv"), row.names = FALSE)

cat("\nExported:\n")
cat("  ", file.path(out_dir, "dax_logreturns.csv"), "(", nrow(dax), "rows )\n")
cat("  ", file.path(out_dir, "dax_2000_2022.csv"), "(full data)\n")
# Also export S&P 500 for sp500_regime_example
cat("\nDownloading ^GSPC (S&P 500, 2000-2022)...\n")
suppressWarnings(
    getSymbols("^GSPC", from = "2000-01-01", to = "2022-12-31",
               auto.assign = TRUE, warnings = FALSE)
)
sp  <- as.data.frame(GSPC)
sp$Date  <- as.Date(rownames(sp))
sp       <- sp[!is.na(sp$GSPC.Close), ]
sp       <- sp[order(sp$Date), ]
sp$logreturn <- c(NA, diff(log(sp$GSPC.Close)))
sp           <- sp[!is.na(sp$logreturn), ]
sp_path <- file.path(out_dir, "sp500_logreturns.csv")
write.csv(data.frame(logreturn = sp$logreturn), sp_path, row.names = FALSE)
cat("Exported", nrow(sp), "S&P 500 log-returns ->", sp_path, "\n")

cat("\nDone. Run: dax_regime_example", out_dir, "\n")
cat("       or: sp500_regime_example", out_dir, "\n")
