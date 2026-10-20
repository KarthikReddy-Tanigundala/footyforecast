#!/usr/bin/env Rscript
# =============================================================================
# FootyForecast — data extraction (the ONLY job R does in the modelling path)
#
#   fitzRoy  ──►  data/raw/*.parquet  +  data/raw/_manifest.json
#
# Usage (from repo root):
#   Rscript R/extract_data.R                 # pull anything not already cached
#   Rscript R/extract_data.R --refresh       # re-pull everything
#   Rscript R/extract_data.R --incremental   # in-season: re-pull the current season only
#   Rscript R/extract_data.R --only=results,odds
#
# Etiquette: Squiggle is a free community service. Requests carry a User-Agent with
# a contact email (FOOTY_CONTACT_EMAIL from .env), whole seasons are pulled once and
# cached, and nothing here runs inside a user-facing request path.
# =============================================================================

suppressPackageStartupMessages({
  library(fitzRoy)
  library(dplyr)
  library(nanoparquet)
  library(jsonlite)
})

# ---- Settings -----------------------------------------------------------------
RAW_DIR         <- file.path("data", "raw")
MANIFEST        <- file.path(RAW_DIR, "_manifest.json")
RESULTS_FROM    <- 1990L   # Elo warm-up starts here
ODDS_FROM       <- 2010L   # Footywire odds history starts in 2010
SQUIGGLE_FROM   <- 2015L   # earliest year to probe for Squiggle tips/games
THIS_YEAR       <- as.integer(format(Sys.Date(), "%Y"))
SEASON_LIVE     <- as.integer(format(Sys.Date(), "%m")) %in% 3:9   # Mar–Sep
LAST_SEASON     <- if (SEASON_LIVE) THIS_YEAR else THIS_YEAR - (as.integer(format(Sys.Date(), "%m")) < 3)
NEXT_SEASON     <- if (SEASON_LIVE) THIS_YEAR else LAST_SEASON + 1L

args        <- commandArgs(trailingOnly = TRUE)
REFRESH     <- "--refresh" %in% args
INCREMENTAL <- "--incremental" %in% args
only_arg    <- sub("^--only=", "", grep("^--only=", args, value = TRUE))
ALL_TARGETS <- c("results", "odds", "squiggle_sources", "squiggle_tips", "squiggle_games", "fixture")
TARGETS     <- if (length(only_arg)) strsplit(only_arg, ",")[[1]] else ALL_TARGETS
REQUIRED    <- c("results")     # a failure here fails the run (CI notices)

if (file.exists(".env")) readRenviron(".env")
CONTACT <- Sys.getenv("FOOTY_CONTACT_EMAIL")
UA <- sprintf(
  "FootyForecast/0.1 (+https://github.com/KarthikReddy-Tanigundala/footyforecast%s)",
  if (nzchar(CONTACT)) paste0("; ", CONTACT) else ""
)
if (!nzchar(CONTACT)) message("! FOOTY_CONTACT_EMAIL not set: Squiggle asks for a contact email in the User-Agent.")

dir.create(RAW_DIR, recursive = TRUE, showWarnings = FALSE)
log_line <- function(...) message(format(Sys.time(), "%H:%M:%S "), sprintf(...))

# ---- Helpers --------------------------------------------------------------------
# Parquet can't hold list columns; Squiggle JSON occasionally produces them.
flatten_df <- function(df) {
  df <- as.data.frame(df)
  is_list <- vapply(df, is.list, logical(1))
  df[is_list] <- lapply(df[is_list], function(col) vapply(col, function(x) paste(unlist(x), collapse = ";"), ""))
  df
}

write_raw <- function(df, name) {
  path <- file.path(RAW_DIR, paste0(name, ".parquet"))
  nanoparquet::write_parquet(flatten_df(df), path)
  path
}

read_raw <- function(name) {
  path <- file.path(RAW_DIR, paste0(name, ".parquet"))
  if (file.exists(path)) as.data.frame(nanoparquet::read_parquet(path)) else NULL
}

is_cached <- function(name) file.exists(file.path(RAW_DIR, paste0(name, ".parquet")))

# Replace one season's rows in an existing extract (incremental mode).
upsert_season <- function(old, new, season_col, seasons) {
  if (is.null(old)) return(new)
  bind_rows(old[!(old[[season_col]] %in% seasons), , drop = FALSE], new)
}

squiggle_years <- function(query, years) {
  frames <- lapply(years, function(y) {
    out <- tryCatch(
      fitzRoy::fetch_squiggle_data(query, year = y, user_agent = UA),
      error = function(e) { log_line("  %s %d: %s", query, y, conditionMessage(e)); NULL }
    )
    Sys.sleep(1)   # be gentle: one request per second
    if (is.null(out) || !nrow(out)) NULL else flatten_df(out)
  })
  bind_rows(frames)
}

# ---- Pulls -----------------------------------------------------------------------
pull <- list(
  results = function(seasons) fitzRoy::fetch_results_afltables(seasons),
  odds = function(seasons) fitzRoy::fetch_betting_odds_footywire(min(seasons), max(seasons)),
  squiggle_sources = function(seasons) fitzRoy::fetch_squiggle_data("sources", user_agent = UA),
  squiggle_tips = function(seasons) squiggle_years("tips", seasons),
  squiggle_games = function(seasons) squiggle_years("games", seasons),
  fixture = function(seasons) fitzRoy::fetch_fixture(max(seasons), source = "AFL")
)

full_window <- list(
  results = RESULTS_FROM:LAST_SEASON,
  odds = ODDS_FROM:LAST_SEASON,
  squiggle_sources = NA,
  squiggle_tips = SQUIGGLE_FROM:LAST_SEASON,
  squiggle_games = SQUIGGLE_FROM:NEXT_SEASON,   # includes next season's fixture once published
  fixture = NEXT_SEASON
)
incremental_window <- list(
  results = LAST_SEASON, odds = LAST_SEASON, squiggle_sources = NA,
  squiggle_tips = LAST_SEASON, squiggle_games = unique(c(LAST_SEASON, NEXT_SEASON)), fixture = NEXT_SEASON
)
season_col <- list(results = "Season", odds = "Season", squiggle_tips = "year", squiggle_games = "year")

manifest <- if (file.exists(MANIFEST)) jsonlite::read_json(MANIFEST) else list(targets = list())
failed_required <- FALSE

for (target in TARGETS) {
  if (!target %in% ALL_TARGETS) stop("Unknown target: ", target)
  if (is_cached(target) && !REFRESH && !INCREMENTAL) {
    log_line("%-17s cached (use --refresh to re-pull)", target)
    next
  }
  incremental <- INCREMENTAL && !is.null(season_col[[target]]) && is_cached(target)
  seasons <- if (incremental) incremental_window[[target]] else full_window[[target]]
  log_line("%-17s pulling %s", target, if (all(is.na(seasons))) "all" else paste(range(seasons), collapse = "–"))

  df <- tryCatch(pull[[target]](seasons), error = function(e) {
    log_line("%-17s FAILED: %s", target, conditionMessage(e))
    NULL
  })
  if (is.null(df) || !nrow(df)) {
    if (target %in% REQUIRED) failed_required <- TRUE
    manifest$targets[[target]]$last_status <- "failed_or_empty"
    next
  }
  if (incremental) df <- upsert_season(read_raw(target), flatten_df(df), season_col[[target]], seasons)

  path <- write_raw(df, target)
  manifest$targets[[target]] <- list(
    file = basename(path),
    rows = nrow(df),
    columns = ncol(df),
    seasons = if (!is.null(season_col[[target]]) && season_col[[target]] %in% names(df))
      as.list(range(df[[season_col[[target]]]], na.rm = TRUE)) else NULL,
    pulled_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"),
    mode = if (incremental) "incremental" else "full",
    last_status = "ok"
  )
  log_line("%-17s wrote %s rows → %s", target, format(nrow(df), big.mark = ","), path)
}

manifest$fitzroy_version <- as.character(packageVersion("fitzRoy"))
manifest$r_version <- R.version.string
manifest$generated_at <- format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z")
jsonlite::write_json(manifest, MANIFEST, auto_unbox = TRUE, pretty = TRUE, null = "null")

if (failed_required) {
  log_line("A required extract failed — exiting non-zero.")
  quit(status = 1)
}
log_line("Extract complete.")
