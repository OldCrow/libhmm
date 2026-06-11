#!/usr/bin/env Rscript
#
# prepare_mv_regime_data.R — Download SPY + QQQ monthly log-returns for mv_regime_example
#
# Downloads SPY (S&P 500 ETF) and QQQ (Nasdaq-100 ETF) monthly closing prices from
# Yahoo Finance (2000-01-01 to 2022-12-31) and writes log-returns to a CSV file.
#
# Usage:
#   Rscript scripts/prepare_mv_regime_data.R [output_dir]
#
# Output files:
#   spy_qqq_monthly.csv   — columns: date, spy_logret, qqq_logret (returns in %)
#
# Then build and run the C++ example:
#   cmake --build build --target mv_regime_example
#   ./build/examples/mv_regime_example [output_dir]
#
# Compare against the Python reference:
#   /tmp/libhmm_hmmlearn_venv/bin/python3 scripts/verify_mv_regime.py [output_dir]
#   (see scripts/verify_mv_regime.py for hmmlearn install instructions)
#
# Requirements:
#   install.packages("quantmod")   # auto-installed if absent
#
# Data source:
#   Yahoo Finance via quantmod. SPY and QQQ are the most liquid large-cap US
#   equity ETFs; their monthly log-returns span the dot-com bust (2000-2002),
#   the Global Financial Crisis (2007-2009), and the COVID crash (2020),
#   providing clear bull / bear / crisis regime structure.
#
# Reference comparison:
#   hmmlearn 0.3.3 GaussianHMM — see scripts/verify_mv_regime.py
#

args    <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) > 0) args[1] else "/tmp"

options(timeout = 120)

if (!require("quantmod", quietly = TRUE)) {
    message("Installing quantmod...")
    install.packages("quantmod", repos = "https://cran.r-project.org", quiet = TRUE)
    library(quantmod, quietly = TRUE)
}

cat("Downloading SPY and QQQ monthly closes from Yahoo Finance (2000-01-01 to 2022-12-31)...\n")
suppressWarnings({
    getSymbols("SPY", src = "yahoo",
               from = "2000-01-01", to = "2022-12-31",
               periodicity = "monthly", auto.assign = TRUE, warnings = FALSE)
    getSymbols("QQQ", src = "yahoo",
               from = "2000-01-01", to = "2022-12-31",
               periodicity = "monthly", auto.assign = TRUE, warnings = FALSE)
})

# Compute monthly log-returns in percent
spy_ret <- na.omit(diff(log(Cl(SPY)))) * 100
qqq_ret <- na.omit(diff(log(Cl(QQQ)))) * 100

# Merge on common dates and drop any remaining NAs
merged  <- na.omit(merge(spy_ret, qqq_ret))
colnames(merged) <- c("spy_logret", "qqq_logret")

cat("Observations:", nrow(merged), "\n")
cat("Date range:", format(start(merged)), "to", format(end(merged)), "\n\n")

cat(sprintf("SPY: mean=%+.3f%%  sd=%.3f%%\n",
            mean(merged$spy_logret), sd(merged$spy_logret)))
cat(sprintf("QQQ: mean=%+.3f%%  sd=%.3f%%\n",
            mean(merged$qqq_logret), sd(merged$qqq_logret)))
cat(sprintf("Overall correlation: %.3f\n\n",
            cor(merged$spy_logret, merged$qqq_logret)))

# Write output
out_path <- file.path(out_dir, "spy_qqq_monthly.csv")
write.csv(data.frame(
    date       = format(index(merged)),
    spy_logret = as.numeric(merged$spy_logret),
    qqq_logret = as.numeric(merged$qqq_logret)
), out_path, row.names = FALSE)

cat("Written:", out_path, "\n")
cat("\nNext steps:\n")
cat("  ./build/examples/mv_regime_example", out_dir, "\n")
cat("  /tmp/libhmm_hmmlearn_venv/bin/python3 scripts/verify_mv_regime.py", out_dir, "\n")
