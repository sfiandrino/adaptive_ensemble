library(hubUtils)
library(dplyr)
library(hubEnsembles)
library(CombineDistributions)
library(jsonlite)
library(yaml)
library(purrr)
library(testthat)

ensemble_lop <- function(df, hor, toptraj, day_tosave, path_tosave, loss_function, is_original, scenario) {
  
  df$horizon <- as.character(df$horizon)
  df$id_var <- paste(df$model_name, df$horizon)
  df$horizon <- as.integer(df$horizon)
  if (is_original == FALSE) {
    df <- df %>% filter(horizon >= hor)
  }
  # list horizons from start_h to end_h
  horizons <- as.list(unique(df$horizon))
  df <- df %>% rename('id' = 'model_name')
  # give half weight to models from the same team
  weights <- data.frame(id = c('CDDEP-FluCompModel', 'MOBS_NEU-GLEAM_FLU', 'NIH-FluD', 'NIH-Flu_TS', 'NotreDame-FRED',
                  'PSI-M2', 'USC-SIkJalpha', 'UT-ImmunoSEIRS', 'UVA-FluXSim'), weight = c(1, 1, 0.5, 0.5, 1, 1 ,1, 1, 1))
  
  user_specified_weights <- function(df){
    weights <- check_weights_df(weights)
    df <- df %>% dplyr::left_join(weights, by = "id")
    return(df)
  }
  check_weights_df <- function(weights){
    if (length(colnames(weights)) != 2) {
        stop("weights not properly specified (ncol != 2)")}
    if(!("weight" %in% colnames(weights))){
        stop("weights not properly specified (weight col not included)")}
    if(!("id" %in% colnames(weights))){
        colnames(weights)[which(colnames(weights)!="weight")] = "id"
    }
    return(weights)
  }
  finaltable <- data.frame()
  for (h in horizons) {
  subdf <- df %>% filter(horizon == h)
  quantiles <- c(0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
               0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
               0.95, 0.975, 0.99)
  lop <- LOP(subdf$quantile, subdf$value, subdf$id, quantiles, NA, weight_fn = user_specified_weights)
  horizon <- rep(h, each = length(quantiles))
  lop_new <- lop %>% mutate('horizon' = horizon)
  finaltable <- bind_rows(finaltable, lop_new)
  }
  #"/home/sfiandrino/PhD_Project/AdaptiveEnsemble_Forecasting_Scenario_tests/AdaptiveEnsemble2_S2_LOP/"
  if (is_original == TRUE) {
    finalpath <- paste0(path_tosave, "Ensemble_", scenario, "_USnational.csv")
  } else {
    finalpath <- paste0(path_tosave, day_tosave, "_", toptraj, "_", loss_function, ".csv")
  }
  
  write.csv(finaltable, finalpath)
  return(finaltable)
}