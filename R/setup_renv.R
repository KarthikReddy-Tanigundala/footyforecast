# One-off: initialise renv and install the pinned R dependencies.
# Usage (from repo root):  Rscript R/setup_renv.R     or   .\make.ps1 r-setup
# Afterwards collaborators only need:  Rscript -e "renv::restore()"

options(repos = c(CRAN = "https://cloud.r-project.org"))

if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")

if (!file.exists("renv.lock")) {
  # Explicit snapshots: only packages declared in DESCRIPTION are locked.
  renv::init(bare = TRUE, settings = list(snapshot.type = "explicit"), restart = FALSE)
}

renv::install(c(
  "fitzRoy@1.8.0", "nanoparquet", "dplyr", "tidyr", "purrr", "lubridate",
  "jsonlite", "ggplot2", "lme4", "broom", "scales", "knitr", "rmarkdown"
), prompt = FALSE)

renv::snapshot(prompt = FALSE)
message("renv ready: ", packageVersion("fitzRoy"), " (fitzRoy)")
