### here do the thing with generate_noise_image
# source("/pyrcicr/rcicr/rcicr/R/generateNoisePattern.R", encoding = "UTF-8")
# source("/pyrcicr/rcicr/rcicr/R/generateSinusoid.R", encoding = "UTF-8")
# source("/pyrcicr/rcicr/rcicr/R/deg2rad.R", encoding = "UTF-8")
generateNoiseImageMinRepro <- function(n_trials=770, img_size=512, noise_type='sinusoid', nscales=5, sigma=25, ncores=3, return_as_dataframe=TRUE) {

  p <- generateNoisePattern(img_size, noise_type=noise_type, nscales=nscales, sigma=sigma)
  # Compute number of parameters needed  #
  nparams <- sum(6*2*(2^(0:(nscales-1)))^2)

  # Generate stimuli parameters, one set for all base faces
  params <- matlab::zeros(n_trials, nparams)
  for (trial in 1:n_trials) {
    params[trial,] <- (runif(nparams) * 2) - 1
  }

  stimuli_params <- params
  
  for (trial in 1:n_trials) {
    stimuli_params[trial,] <- (runif(nparams) * 2) - 1
  }  

  # Generate stimuli
  pb <- txtProgressBar(min = 1, max = n_trials, style = 3)

  stimuli <- matlab::zeros(img_size, img_size, n_trials)

  cl <- parallel::makeCluster(ncores, outfile = "")
  doParallel::registerDoParallel(cl)

  stims <- foreach::foreach(
    trial = 1:n_trials, .packages = 'rcicr', .final = function(x) setNames(as.data.frame(x), as.character(1:n_trials)), .combine = 'cbind', .multicombine = TRUE) %dopar% {
    stimuli[,,trial] <- generateNoiseImage(stimuli_params[trial,], p)
    if (return_as_dataframe) {
      return(as.vector(stimuli[,,trial]))
    }

    # Update progress bar
    setTxtProgressBar(pb, trial)
  }
  parallel::stopCluster(cl)
  # Return CIs
  if (return_as_dataframe) {
    return(stims)
  }
}

# generateNoiseImageMinRepro()