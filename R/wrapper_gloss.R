#' Inference for Global Local Spatial Selection with varying coefficients
#'
#' @param y vector of observations (response variable)
#' @param X design matrix (\code{n x p})
#' @param hyper list of hyperparameters (see paper for details)
#' @param fix_delta if 1 varying coefficients are considered (V-LoSS); if 0 constant coefficients are considered (C-LoSS); default is NULL, that is GLoSS is estimated
#' @param local if FALSE no local selection is performed; default is TRUE
#' @param R number of posterior draws
#' @param B number of burn-in draws. Final number of draws is \code{R-B}
#' @param Trace if 1, print progress of the algorithm
#' 
#'
#' @return  Samples from posterior distributions according to the selected method (see the paper for details, the names of the variables match)
#' @export
GLoSS = function(y,X,hyper,fix_delta=NULL,local=TRUE,R=2000,B=1000,Trace=0) {
  if (is.null(fix_delta)) mod = gloss(y,X,hyper,R,B,Trace)
  
  if (!is.null(fix_delta)) {
    if (fix_delta == 1) mod = vloss(y,X,hyper,R,B,Trace)
    if (fix_delta == 0) mod = closs(y,X,hyper,R,B,Trace)
  }
  
  if (!local) mod = spgp(y,X,hyper,R,B,Trace)
  
  mod
}


#' Spatial prediction for Global Local Spatial Selection with varying coefficients
#'
#' @param j index of the coefficient to predict
#' @param mod output of GLoSS function
#' @param prParam list of elicited hyperparameters (see paper for details)
#' @param iloc_test (\code{n_test x 2}) matrix with the new locations
#' @param iloc_train (\code{n_train x 2}) matrix with the locations used for estimation
#' @param setting one of the special cases GLoSS, V-LoSS, C-LoSS, GP; default is GLoSS
#' 
#'
#' @return  Samples from posterior predictive distributions of the j-th coefficient and local indicator
#' @export
predictBetaGamma = function(j,mod,prParam,iloc_test,iloc_train,setting='GLoSS') {
  
  if (setting=='GLoSS') {
    Rmix1 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rmix0 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[4])
    
    Rnew1 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rnew0 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[4])
    
    Rom_mix = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prGam[2])
    Rom_new = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prGam[2])
    
    krigBeta = krig(mod$beta[,j,],R0,R1,Rmix0,Rmix1,Rnew0,Rnew1,mod$PIP_delta[,j],mod$mu[,j])
    
    krigOm = krig_gamma(mod$omega[,j,],ROm,Rom_mix,Rom_new,mod$m[,j])
    krigPi = plogis(krigOm)
    
    krGammaBeta = krigBeta*krigPi
    
    out = list(Gamma=krigPi,GammaBeta=krGammaBeta)
  }
  
  if (setting=='VLoSS') {
    n0 = dim(mod$beta)[3]
    
    Rmix1 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rmix0 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[4])
    
    Rnew1 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rnew0 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[4])
    
    Rom_mix = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prGam[2])
    Rom_new = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prGam[2])
    
    krigBeta = krig(mod$beta[,j,],R0,R1,Rmix0,Rmix1,Rnew0,Rnew1,rep(1,n0),mod$mu[,j])
    
    krigOm = krig_gamma(mod$omega[,j,],ROm,Rom_mix,Rom_new,mod$m[,j])
    krigPi = plogis(krigOm)
    
    krGammaBeta = krigBeta*krigPi
    
    out = list(Gamma=krigPi,GammaBeta=krGammaBeta)
  }
  
  if (setting=='CLoSS') {
    n0 = dim(mod$beta)[3]
    
    Rmix1 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rmix0 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=100)
    
    Rnew1 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rnew0 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=100)
    
    Rom_mix = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prGam[2])
    Rom_new = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prGam[2])
    
    krigBeta = krig(mod$beta[,j,],R0,R1,Rmix0,Rmix1,Rnew0,Rnew1,rep(0,n0),mod$mu[,j])
    
    krigOm = krig_gamma(mod$omega[,j,],ROm,Rom_mix,Rom_new,mod$m[,j])
    krigPi = plogis(krigOm)
    
    krGammaBeta = krigBeta*krigPi
    
    out = list(Gamma=krigPi,GammaBeta=krGammaBeta)
  }
  
  if (setting=='GP') {
    n0 = dim(mod$beta)[3]
    
    Rmix1 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rmix0 = stationary.cov(iloc_test,iloc_train,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[4])
    
    Rnew1 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[3])
    Rnew0 = stationary.cov(iloc_test,iloc_test,Covariance = "Matern",smoothness=0.5,theta=prParam$prBeta[4])
    
    krigBeta = krig(mod$beta[,j,],R0,R1,Rmix0,Rmix1,Rnew0,Rnew1,rep(1,n0),mod$mu[,j])
    
    out = list(Beta=krigBeta)
  }
  
  
  out
}
