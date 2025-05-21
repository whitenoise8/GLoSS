#-------  Prior Scaling | Functions -------#

#' Prior elicitation for Global Local Spatial Selection with varying coefficients
#'
#' @param parGrid0 grid of parameters \eqn{(k, \phi_{\beta,0})} for the global selection when \eqn{\delta}=0
#' @param t01 threshold for W_1 when \eqn{\delta}=0
#' @param t02 threshold for W_2 when \eqn{\delta}=0
#' @param alpha0 probability of event when \eqn{\delta}=0
#' @param parGrid1 grid of parameters \eqn{(b_\eta, \phi_{\beta,1})} for the global selection when \eqn{\delta}=1
#' @param t11 threshold for W_1 when \eqn{\delta}=1
#' @param t12 threshold for W_2 when \eqn{\delta}=1
#' @param alpha1 probability of event when \eqn{\delta}=1
#' @param parGridOm grid of parameters \eqn{(\xi^2, \phi_{\omega})} for the local selection
#' @param iloc (\code{n x 2}) matrix with the locations
#' @param Ae prior parameter for \eqn{\eta^2\sim IG(a_\eta,b_\eta)}; default 2
#' @param nu smoothness of spatial correlation kernel; default 0.5
#' @param N number of draws for the simulation-based elicitation; default 1000
#' 
#'
#' @return  Elicited prior parameters and empirical probabilities (see paper for more details)
#' @return  \code{prBeta}: a vector with \eqn{(b_\eta, k, \phi_{\beta,1}, \phi_{\beta,0}, \hat{\alpha}_0,\hat{\alpha}_1)}  
#' @return  \code{prGam}: a vector with \eqn{(\xi^2, k, \phi_{\omega}, \hat{p}_s, \hat{p}_{\rho|s})}  

#' @export
elicitPrior = function(parGrid0,t01,t02,alpha0,parGrid1,t11,t12,alpha1,parGridOm,iloc,Ae=2,nu=0.5,N=1000) {
  
  # Matrix of distances
  H = apply(iloc,1,function(y) apply(iloc,1,function(x) EuclDist2D(x,y)))
  
  # Prior scaling
  prPar = priorScaling(parGrid0,t01,t02,alpha0,parGrid1,t11,t12,alpha1,Ae,nu,iloc_train,H,N)
  prParGam = priorScalingGamma(parGridOm,iloc_train,nu,tau2,N)
  
  list(prBeta=prPar,prGam=prParGam)
}

priorScaling = function(parGrid0,t01,t02,alpha0,parGrid1,t11,t12,alpha1,Ae,nu,iloc,H,N) {
  require(mvnfast)
  require(fields)
  
  n = nrow(H)
  Nei = apply(matrix(1:n), 1, function(x) sort.int(H[x,],index.return=T)$ix[2])
  
  pr0 = pr1 = prOm = rep(NA,nrow(parGrid1))
  
  for (i in 1:nrow(parGrid1)) {
    Be0 = parGrid0[i,1]
    th0 = parGrid0[i,2]
    Be1 = parGrid1[i,1]
    th1 = parGrid1[i,2]

    R0 = stationary.cov(iloc,Covariance = "Matern",smoothness=nu,theta=th0)
    R1 = stationary.cov(iloc,Covariance = "Matern",smoothness=nu,theta=th1)
    
    b1 = rmvt(N,mu=rep(0,n),Be1/Ae*R1,2*Ae)
    b_bar1 = apply(b1,1,function(x) max(abs(x)))  
    
    mDiff1 = apply(matrix(1:n), 1, function(x) abs(b1[,x] - b1[,Nei[x]]))
    b_diff1 = apply(mDiff1, 1, max)
    
    pr1[i] = mean((b_bar1 > t11) & (b_diff1 > t12))
    
    b0 = rmvt(N,mu=rep(0,n),Be0/Ae*R0,2*Ae)
    b_bar0 = apply(b0,1,function(x) max(abs(x)))  
    
    mDiff0 = apply(matrix(1:n), 1, function(x) abs(b0[,x] - b0[,Nei[x]]))
    b_diff0 = apply(mDiff0, 1, max)
    
    pr0[i] = mean((b_bar0 < t01) & (b_diff0 < t02))
  }
  
  id0 = which.min(abs(pr0-alpha0))[1]
  id1 = which.min(abs(pr1-alpha1))[1]
  
  Be0 = parGrid0[id0,1]
  th0 = parGrid0[id0,2]
  
  Be1 = parGrid1[id1,1]
  th1 = parGrid1[id1,2]
  
  c(Be1,Be0/Be1,th1,th0,pr0[id0],pr1[id1])
}


trapint = function(xgrid, fgrid) {
  ng = length(xgrid)
  xvec = xgrid[2:ng] - xgrid[1:(ng - 1)]
  fvec = fgrid[1:(ng - 1)] + fgrid[2:ng]
  integ = sum(xvec * fvec)/2
  return(integ)
}


IAE = function(sample,A=1,B=1) {
  require("KernSmooth")
  
  gridSize = 1001
  xg = seq(0,1,length=gridSize) 
  
  VApostg = dbeta(xg,A,B)
  MCpostg = bkde(sample,bandwidth=dpik(sample),range.x=c(0,1),gridsize=gridSize)$y
  
  round(100-50*trapint(xg,abs(MCpostg-VApostg)),2)/100
}


priorScalingGamma = function(parGrid,iloc,nu,tau2,N,thresh=0.8) {
  require(mvnfast)
  require(fields)
  
  n = nrow(iloc)
  A = B = rep(NA,nrow(parGrid))
  
  for (i in 1:nrow(parGrid)) {
    xi2 = parGrid[i,1]
    phi = parGrid[i,2]
    
    R = xi2*stationary.cov(iloc,Covariance = "Matern",smoothness=nu,theta=phi)
    
    om = rmvn(N,mu=rep(0,n),R+tau2*matrix(1,n,n))
    pi = plogis(om)
    gamma = apply(pi,1,function(x) rbinom(n,1,x))
    
    s = colMeans(gamma)
    A[i] = IAE(s)
    
    id = (s > 0.4)&(s < 0.6)
    gamma = gamma[,id]
    
    PrRho = apply(gamma,2,function(gam) {
      
      R11 = R[gam==1,gam==1]/xi2
      R10 = R[gam==1,gam==0]/xi2
      
      gamma0 = rep(0,n)
      gamma0[1:sum(gam)] = 1
      gamma0 = sample(gamma0,n)
      
      R01 = R[gamma0==1,gamma0==1]/xi2
      R00 = R[gamma0==1,gamma0==0]/xi2
      
      r = (mean(R11[lower.tri(R11)]) - mean(R10[lower.tri(R10)]))/abs(mean(R11[lower.tri(R11)])-mean(R01[lower.tri(R01)]))
      r > (2-mean(gam))/(2*(1-mean(gam)))
      
    })
    
    B[i] = mean(PrRho)
  }
  
  id = which.max(B*A)
  c(as.numeric(parGrid[id,]),A[id],B[id])
}


