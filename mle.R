library(ismev)
library(RcppCNPy)
library(parallel)

compute_ci_width <- function(x) {
  out <- tryCatch({
    fit <- gev.fit(x, show = FALSE)
    
    if (is.null(fit$cov)) {
      return(c(mu = NA_real_, sigma = NA_real_, xi = NA_real_))
    }
    
    est <- fit$mle
    v <- diag(fit$cov)
    
    if (any(!is.finite(est)) || any(!is.finite(v)) || any(v < 0)) {
      return(c(mu = NA_real_, sigma = NA_real_, xi = NA_real_))
    }
    
    se <- sqrt(v)
    
    if (any(!is.finite(se))) {
      return(c(mu = NA_real_, sigma = NA_real_, xi = NA_real_))
    }
    
    lower <- est - 1.96 * se
    upper <- est + 1.96 * se
    width <- upper - lower
    
    c(
      mu    = unname(width[1]),
      sigma = unname(width[2]),
      xi    = unname(width[3])
    )
  }, error = function(e) {
    c(mu = NA_real_, sigma = NA_real_, xi = NA_real_)
  })
  
  return(out)
}

meta <- read.csv(
  "C:/Users/User.DESKTOP-4RV84M1/Desktop/fast_parameter_using_NN/sample_index.csv",
  stringsAsFactors = FALSE
)

meta$file_path <- gsub("\\\\", "/", meta$file_path)

n_cores <- 6
cat("Using", n_cores, "cores\n")

cl <- makeCluster(n_cores)

clusterEvalQ(cl, {
  library(ismev)
  library(RcppCNPy)
})

clusterExport(
  cl,
  varlist = c("meta", "compute_ci_width"),
  envir = environment()
)

results_list <- parLapply(cl, seq_len(nrow(meta)), function(i) {
  file_path <- meta$file_path[i]
  x <- npyLoad(file_path)
  
  w <- compute_ci_width(x)
  
  data.frame(
    panel = meta$panel[i],
    true_value = meta$true_value[i],
    rep = meta$rep[i],
    ml_width_mu = w["mu"],
    ml_width_sigma = w["sigma"],
    ml_width_xi = w["xi"]
  )
})

stopCluster(cl)

results <- do.call(rbind, results_list)

write.csv(
  results,
  "C:/Users/User.DESKTOP-4RV84M1/Desktop/fast_parameter_using_NN/mle_results.csv",
  row.names = FALSE
)

cat("Saved mle_results.csv\n")
print(head(results))

