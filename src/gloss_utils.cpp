// [[Rcpp::depends(RcppEigen)]]
// [[Rcpp::depends(RcppNumerical)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <math.h>  
#include <RcppNumerical.h>
#include <cmath>
#include <iostream>

using namespace Numer;
using namespace std;
using namespace arma;
using namespace Rcpp;

// Mathematical constants computed using Wolfram Alpha
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

// FCN prototypes
double samplepg(double);
double exprnd(double);
double tinvgauss(double, double);
double truncgamma();
double randinvg(double);
double aterm(int, double, double);

arma::vec rcpp_pgdraw(arma::vec cc)
{
  NumericVector b(1);
  b[0] = 1;
  int m = 1;
  
  int n = cc.n_elem;
  NumericVector c(n);
  for (int i = 0; i < n; i++)
  {
    c[i] = cc(i);
  }
  
  NumericVector y(n);
  
  // Setup
  int i, j, bi = 1;
  if (m == 1)
  {
    bi = b[0];
  }
  
  // Sample
  for (i = 0; i < n; i++)
  {
    if (m > 1)
    {
      bi = b[i];
    }
    
    // Sample
    y[i] = 0;
    for (j = 0; j < (int)bi; j++)
    {
      y[i] += samplepg(c[i]);
    }
  }
  
  arma::vec yy = y;
  return yy;
}


// Sample PG(1,z)
// Based on Algorithm 6 in PhD thesis of Jesse Bennett Windle, 2013
// URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
double samplepg(double z)
{
  //  PG(b, z) = 0.25 * J*(b, z/2)
  z = (double)std::fabs((double)z) * 0.5;
  
  // Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
  double t = MATH_2_PI;
  
  // Compute p, q and the ratio q / (q + p)
  // (derived from scratch; derivation is not in the original paper)
  double K = z*z/2.0 + MATH_PI2/8.0;
  double logA = (double)std::log(4.0) - MATH_LOG_PI - z;
  double logK = (double)std::log(K);
  double Kt = K * t;
  double w = (double)std::sqrt(MATH_PI_2);
  
  double logf1 = logA + R::pnorm(w*(t*z - 1),0.0,1.0,1,1) + logK + Kt;
  double logf2 = logA + 2*z + R::pnorm(-w*(t*z+1),0.0,1.0,1,1) + logK + Kt;
  double p_over_q = (double)std::exp(logf1) + (double)std::exp(logf2);
  double ratio = 1.0 / (1.0 + p_over_q); 
  
  double u, X;
  
  // Main sampling loop; page 130 of the Windle PhD thesis
  while(1) 
  {
    // Step 1: Sample X ? g(x|z)
    u = R::runif(0.0,1.0);
    if(u < ratio) {
      // truncated exponential
      X = t + exprnd(1.0)/K;
    }
    else {
      // truncated Inverse Gaussian
      X = tinvgauss(z, t);
    }
    
    // Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U ? Sn(X|z) for an odd n or U > Sn(X|z) for an even n
    int i = 1;
    double Sn = aterm(0, X, t);
    double U = R::runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;
    
    while(1) 
    {
      Sn = Sn + asgn * aterm(i, X, t);
      
      // Accept if n is odd
      if(!even && (U <= Sn)) {
        X = X * 0.25;
        return X;
      }
      
      // Return to step 1 if n is even
      if(even && (U > Sn)) {
        break;
      }
      
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
  return X;
}

// Generate exponential distribution random variates
double exprnd(double mu)
{
  return -mu * (double)std::log(1.0 - (double)R::runif(0.0,1.0));
}

// Function a_n(x) defined in equations (12) and (13) of
// Bayesian inference for logistic models using Polya-Gamma latent variables
// Nicholas G. Polson, James G. Scott, Jesse Windle
// arXiv:1205.0310
//
// Also found in the PhD thesis of Windle (2013) in equations
// (2.14) and (2.15), page 24
double aterm(int n, double x, double t)
{
  double f = 0;
  if(x <= t) {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) + 1.5*(MATH_LOG_2_PI- (double)std::log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
  }
  else {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }    
  return (double)exp(f);
}

// Generate inverse gaussian random variates
double randinvg(double mu)
{
  // sampling
  double u = R::rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - (double)std::sqrt(4.0*mu*V + mu*mu * V*V) );
  
  if(R::runif(0.0,1.0) > mu /(mu+out)) {    
    out = mu*mu / out; 
  }    
  return out;
}

// Sample truncated gamma random variates
// Ref: Chung, Y.: Simulation of truncated gamma variables 
// Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
double truncgamma()
{
  double c = MATH_PI_2;
  double X, gX;
  
  bool done = false;
  while(!done)
  {
    X = exprnd(1.0) * 2.0 + c;
    gX = MATH_SQRT_PI_2 / (double)std::sqrt(X);
    
    if(R::runif(0.0,1.0) <= gX) {
      done = true;
    }
  }
  
  return X;  
}

// Sample truncated inverse Gaussian random variates
// Algorithm 4 in the Windle (2013) PhD thesis, page 129
double tinvgauss(double z, double t)
{
  double X, u;
  double mu = 1.0/z;
  
  // Pick sampler
  if(mu > t) {
    // Sampler based on truncated gamma 
    // Algorithm 3 in the Windle (2013) PhD thesis, page 128
    while(1) {
      u = R::runif(0.0, 1.0);
      X = 1.0 / truncgamma();
      
      if ((double)std::log(u) < (-z*z*0.5*X)) {
        break;
      }
    }
  }  
  else {
    // Rejection sampler
    X = t + 1.0;
    while(X >= t) {
      X = randinvg(mu);
    }
  }    
  return X;
}


double log_dmvn(arma::vec x, arma::mat s) {
  double d = x.n_elem;
  double out = -0.5*d*log(2*3.14) -0.5*log_det_sympd(s) -0.5*as_scalar(x.t()*inv_sympd(s)*x);
  return out;
}

double dlog_dmvt(arma::vec x, double mu, arma::mat Sinv1, arma::mat Sinv0, double nu) {
  double d = x.n_elem;
  double out = 0.5*log_det_sympd(Sinv1)-0.5*log_det_sympd(Sinv0)-0.5*(nu+d)*log(1+1.0/nu*as_scalar((x-mu).t()*Sinv1*(x-mu)))+0.5*(nu+d)*log(1+1.0/nu*as_scalar((x-mu).t()*Sinv0*(x-mu)));
  return out;
}

double dlog_dbern(arma::vec x, arma::vec om, double m) {
  double d = x.n_elem;
  double out = sum(x%om) - sum(log(1+exp(om))) - sum(x*m) + d*log(1+exp(m));
  return out;
}


arma::vec rbern(arma::vec p) {
  
  double n = p.n_elem;
  arma::vec x = zeros(n);
  for (int i = 0; i < n; i++) x(i) = R::rbinom(1,p(i));
  
  return x;
}

double log_dexp(double x, double lam){  
  double res = log(lam) -x*lam;
  return(res);
}


// [[Rcpp::export]]
double EuclDist2D(arma::rowvec x, arma::rowvec y) {
  double E2dist = sqrt((x(0)-y(0))*(x(0)-y(0)) + (x(1)-y(1))*(x(1)-y(1)));
  return E2dist;
}

// [[Rcpp::export]]
arma::mat krig(arma::mat beta, 
               arma::mat R0, arma::mat R1, 
               arma::mat Rmix0, arma::mat Rmix1, 
               arma::mat Rnew0, arma::mat Rnew1, 
               arma::vec delta, arma::vec mu) {
  double R = beta.n_cols;
  double n = Rnew0.n_rows;
  
  arma::mat X = zeros(n,R);
  
  for (int r = 0; r < R; r++) {
    arma::mat R0inv = inv_sympd(delta(r)*R1 + (1-delta(r))*R0);
    arma::mat R01 = delta(r)*Rmix1 + (1-delta(r))*Rmix0;
    arma::mat R11 = delta(r)*Rnew1 + (1-delta(r))*Rnew0;
    
    X.col(r) = mu(r) + R01 * R0inv * (beta.col(r) - mu(r));
  }
  
  return X;
}


// [[Rcpp::export]]
arma::mat krig_gamma(arma::mat beta,arma::mat R0,arma::mat Rmix0,arma::mat Rnew0,arma::vec mu) {
  double R = beta.n_cols;
  double n = Rnew0.n_rows;
  
  arma::mat R0inv = inv_sympd(R0);
  arma::mat RR = Rmix0 * R0inv;
  
  arma::mat X = zeros(n,R);
  
  for (int r = 0; r < R; r++) X.col(r) = mu(r) + RR * (beta.col(r) - mu(r));
  
  return X;
}



// [[Rcpp::export]]
Rcpp::List gloss(arma::vec y, arma::mat X, Rcpp::List hyper,
                 int R = 2000, int B = 1000, int Trace = 0) {
  
  double n = X.n_rows;
  double p = X.n_cols;
  
  arma::mat X2 = X%X;
  
  arma::vec beta_0 = inv_sympd(X.t()*X+0.05*eye(p,p))*X.t()*y;
  arma::mat B0 = zeros(n,p);
  for (int t = 0; t < n; t++) B0.row(t) = beta_0.t();
  
  double As = hyper["As"];
  double Bs = hyper["Bs"];
  
  double Ae = hyper["Ae"];
  double Be = hyper["Be"];
  double k  = hyper["k"];
  
  arma::mat R_beta0  = hyper["R0"];
  arma::mat R_beta1  = hyper["R1"];
  
  double tau2 = hyper["tau2"];
  double p0 = hyper["p0"];
  
  double xi2 = hyper["xi2"];
  arma::mat R_om  = hyper["ROm"];
  
  arma::mat Q_beta0 = inv_sympd(R_beta0);
  arma::mat Q_beta1 = inv_sympd(R_beta1);
  arma::mat Sinv_beta1 = Ae/Be*Q_beta1;
  arma::mat Sinv_beta0 = Ae/Be*Q_beta0/k;
  
  arma::mat Q_om = inv_sympd(R_om);
  
  arma::vec sigma2 = ones(R);
  
  arma::cube beta = zeros(n,p,R);
  beta.slice(0) = B0;
  arma::cube gamma = ones(n,p,R);
  arma::cube PIP_gamma = ones(n,p,R);
  arma::cube gamma_beta = beta;
  arma::cube omega = zeros(n,p,R);
  
  arma::mat eta2 = ones(R,p);
  arma::cube z = 0.1*ones(n,p,R);
  
  arma::mat mu = zeros(R,p);
  mu.row(0) = beta_0.t();
  arma::mat m = zeros(R,p);
  arma::mat delta = ones(R,p);
  arma::mat PIP_delta = ones(R,p);
  
  arma::vec printId = regspace(1000, 1000, R);
  
  for (int r = 1; r < R; r++) {
    
    beta.slice(r) = beta.slice(r-1);
    gamma.slice(r) = gamma.slice(r-1);
    gamma_beta.slice(r) = gamma_beta.slice(r-1);
    omega.slice(r)  = omega.slice(r-1);
    z.slice(r)  = z.slice(r-1);
    
    sigma2(r) = sigma2(r-1);
    
    eta2.row(r) = eta2.row(r-1);
    mu.row(r) = mu.row(r-1);
    m.row(r) = m.row(r-1);
    delta.row(r) = delta.row(r-1);
    
    arma::uvec Ord = randperm(p);
    
    for (int indJ = 0; indJ < p; indJ++) {
      double j = Ord(indJ);
      
      arma::vec res = y - sum(X%gamma.slice(r)%beta.slice(r),1) + X.col(j)%gamma.slice(r).col(j)%beta.slice(r).col(j);
      arma::mat Q_beta = delta(r,j)*Q_beta1 + (1-delta(r,j))*Q_beta0;
      double iQi = as_scalar(ones(n).t()*Q_beta*ones(n));
      
      arma::mat Om_beta = 1/sigma2(r)*diagmat(gamma.slice(r).col(j)%X2.col(j)) + 1/eta2(r,j)*Q_beta;
      arma::mat Sigma_beta = inv_sympd(Om_beta);
      arma::vec mu_beta = Sigma_beta*(1/sigma2(r)*gamma.slice(r).col(j)%X.col(j)%res + 1/eta2(r,j)*Q_beta*ones(n)*mu(r,j));  
      beta.slice(r).col(j) = mvnrnd(mu_beta,Sigma_beta,1);
      
      double sigma2_mu = 1/(iQi/eta2(r,j) + 1/tau2);
      double mu_mu = sigma2_mu*as_scalar(1/eta2(r,j)*beta.slice(r).col(j).t()*Q_beta*ones(n));
      mu(r,j) = Rcpp::rnorm(1,mu_mu,sqrt(sigma2_mu))(0);
      
      double A_eta = Ae + 0.5*n;
      double B_eta = Be*(delta(r,j)+(1-delta(r,j))*k) + 0.5*as_scalar((beta.slice(r).col(j)-mu(r,j)).t()*Q_beta*(beta.slice(r).col(j)-mu(r,j))); 
      eta2(r,j) = 1/rgamma(1,A_eta,1/B_eta)(0);
      
      
      arma::mat Om_om = diagmat(z.slice(r).col(j)) + 1/xi2*Q_om;
      arma::mat Sigma_om = inv_sympd(Om_om);
      arma::vec mu_om = Sigma_om*(gamma.slice(r).col(j)-0.5 + 1/xi2*Q_om*ones(n)*m(r,j));
      omega.slice(r).col(j) = mvnrnd(mu_om,Sigma_om,1);
      
      double sigma2_m = 1/(as_scalar(ones(n).t()*Q_om*ones(n))/xi2 + 1/tau2);
      double mu_m = sigma2_m*as_scalar(1/xi2*omega.slice(r).col(j).t()*Q_om*ones(n));
      m(r,j) = Rcpp::rnorm(1,mu_m,sqrt(sigma2_m))(0);
      
      z.slice(r).col(j) = rcpp_pgdraw(omega.slice(r).col(j));
      
      arma::vec omega_q = omega.slice(r).col(j)-0.5/sigma2(r)*(pow(beta.slice(r).col(j),2)%X2.col(j) - 2*beta.slice(r).col(j)%X.col(j)%res);
      omega_q(find(abs(omega_q)>50)) = sign(omega_q(find(abs(omega_q)>50)))*50;
      gamma.slice(r).col(j) = rbern(exp(omega_q)/(1+exp(omega_q)));
      PIP_gamma.slice(r).col(j) = exp(omega_q)/(1+exp(omega_q));
      
      
      double p_q = dlog_dmvt(beta.slice(r).col(j), mu(r,j), Sinv_beta1, Sinv_beta0, 2*Ae) + log(p0)-log(1-p0);
      if (abs(p_q)>50) p_q = sign(p_q)*50;
      delta(r,j) = R::rbinom(1,exp(p_q)/(1+exp(p_q)));
      PIP_delta(r,j) = exp(p_q)/(1+exp(p_q));
      
      if (sum(gamma.slice(r).col(j)) == 0) delta(r,j) = 0;
      
    }
    
    gamma_beta.slice(r) = gamma.slice(r)%beta.slice(r);
    
    double A_s2 = As + 0.5*n;
    double B_s2 = Bs + 0.5*sum((y-sum(X%gamma_beta.slice(r),1))%(y-sum(X%gamma_beta.slice(r),1)));
    sigma2(r) = 1/rgamma(1,A_s2,1/B_s2)(0);
    
    if (Trace == 1) {
      if (size(find(printId == r+1))(0) == 1) Rcout << "Iteration " << r+1 << " of " << R << endl;
    }
  }
  
  beta.shed_slices(0,B-1);
  gamma.shed_slices(0,B-1);
  gamma_beta.shed_slices(0,B-1);
  omega.shed_slices(0,B-1);
  eta2.shed_rows(0,B-1);
  mu.shed_rows(0,B-1);
  m.shed_rows(0,B-1);
  delta.shed_rows(0,B-1);
  sigma2.shed_rows(0,B-1);
  PIP_gamma.shed_slices(0,B-1);
  PIP_delta.shed_rows(0,B-1);
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("gamma") = gamma,
    Rcpp::Named("PIP_gamma") = PIP_gamma,
    Rcpp::Named("gamma_beta") = gamma_beta,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("eta2") = eta2,
    Rcpp::Named("mu") = mu,
    Rcpp::Named("m") = m,
    Rcpp::Named("delta") = delta,
    Rcpp::Named("PIP_delta") = PIP_delta,
    Rcpp::Named("sigma2") = sigma2
  ); 
}


// [[Rcpp::export]]
Rcpp::List closs(arma::vec y, arma::mat X, Rcpp::List hyper,
                 int R = 2000, int B = 1000, int Trace = 0) {
  
  double n = X.n_rows;
  double p = X.n_cols;
  
  arma::mat X2 = X%X;
  
  arma::vec beta_0 = inv_sympd(X.t()*X+0.05*eye(p,p))*X.t()*y;
  arma::mat B0 = zeros(n,p);
  for (int t = 0; t < n; t++) B0.row(t) = beta_0.t();
  
  double As = hyper["As"];
  double Bs = hyper["Bs"];
  
  double Ae = hyper["Ae"];
  double Be = hyper["Be"];
  double k  = hyper["k"];
  
  arma::mat R_beta  = hyper["R0"];
  
  double tau2 = hyper["tau2"];
  double p0 = hyper["p0"];
  
  double xi2 = hyper["xi2"];
  arma::mat R_om  = hyper["ROm"];
  
  arma::mat Q_beta = inv_sympd(R_beta);
  double iQi = as_scalar(ones(n).t()*Q_beta*ones(n));
  
  arma::mat Q_om = inv_sympd(R_om);
  
  arma::vec sigma2 = ones(R);
  
  arma::cube beta = zeros(n,p,R);
  beta.slice(0) = B0;
  arma::cube gamma = ones(n,p,R);
  arma::cube PIP_gamma = ones(n,p,R);
  arma::cube gamma_beta = beta;
  arma::cube omega = zeros(n,p,R);
  
  arma::mat eta2 = ones(R,p);
  arma::cube z = 0.1*ones(n,p,R);
  
  arma::mat mu = zeros(R,p);
  mu.row(0) = beta_0.t();
  arma::mat m = zeros(R,p);
  
  arma::vec printId = regspace(1000, 1000, R);
  
  for (int r = 1; r < R; r++) {
    
    beta.slice(r) = beta.slice(r-1);
    gamma.slice(r) = gamma.slice(r-1);
    gamma_beta.slice(r) = gamma_beta.slice(r-1);
    omega.slice(r)  = omega.slice(r-1);
    z.slice(r)  = z.slice(r-1);
    
    sigma2(r) = sigma2(r-1);
    
    eta2.row(r) = eta2.row(r-1);
    mu.row(r) = mu.row(r-1);
    m.row(r) = m.row(r-1);
    
    arma::uvec Ord = randperm(p);
    
    for (int indJ = 0; indJ < p; indJ++) {
      double j = Ord(indJ);
      
      arma::vec res = y - sum(X%gamma.slice(r)%beta.slice(r),1) + X.col(j)%gamma.slice(r).col(j)%beta.slice(r).col(j);
      
      arma::mat Om_beta = 1/sigma2(r)*diagmat(gamma.slice(r).col(j)%X2.col(j)) + 1/eta2(r,j)*Q_beta;
      arma::mat Sigma_beta = inv_sympd(Om_beta);
      arma::vec mu_beta = Sigma_beta*(1/sigma2(r)*gamma.slice(r).col(j)%X.col(j)%res + 1/eta2(r,j)*Q_beta*ones(n)*mu(r,j));  
      beta.slice(r).col(j) = mvnrnd(mu_beta,Sigma_beta,1);
      
      double sigma2_mu = 1/(iQi/eta2(r,j) + 1/tau2);
      double mu_mu = sigma2_mu*as_scalar(1/eta2(r,j)*beta.slice(r).col(j).t()*Q_beta*ones(n));
      mu(r,j) = Rcpp::rnorm(1,mu_mu,sqrt(sigma2_mu))(0);
      
      double A_eta = Ae + 0.5*n;
      double B_eta = Be*k + 0.5*as_scalar((beta.slice(r).col(j)-mu(r,j)).t()*Q_beta*(beta.slice(r).col(j)-mu(r,j))); 
      eta2(r,j) = 0.001;
      
      
      arma::mat Om_om = diagmat(z.slice(r).col(j)) + 1/xi2*Q_om;
      arma::mat Sigma_om = inv_sympd(Om_om);
      arma::vec mu_om = Sigma_om*(gamma.slice(r).col(j)-0.5 + 1/xi2*Q_om*ones(n)*m(r,j));
      omega.slice(r).col(j) = mvnrnd(mu_om,Sigma_om,1);
      
      double sigma2_m = 1/(as_scalar(ones(n).t()*Q_om*ones(n))/xi2 + 1/tau2);
      double mu_m = sigma2_m*as_scalar(1/xi2*omega.slice(r).col(j).t()*Q_om*ones(n));
      m(r,j) = Rcpp::rnorm(1,mu_m,sqrt(sigma2_m))(0);
      
      z.slice(r).col(j) = rcpp_pgdraw(omega.slice(r).col(j));
      
      arma::vec omega_q = omega.slice(r).col(j)-0.5/sigma2(r)*(pow(beta.slice(r).col(j),2)%X2.col(j) - 2*beta.slice(r).col(j)%X.col(j)%res);
      omega_q(find(abs(omega_q)>50)) = sign(omega_q(find(abs(omega_q)>50)))*50;
      gamma.slice(r).col(j) = rbern(exp(omega_q)/(1+exp(omega_q)));
      PIP_gamma.slice(r).col(j) = exp(omega_q)/(1+exp(omega_q));
      
    }
    
    gamma_beta.slice(r) = gamma.slice(r)%beta.slice(r);
    
    double A_s2 = As + 0.5*n;
    double B_s2 = Bs + 0.5*sum((y-sum(X%gamma_beta.slice(r),1))%(y-sum(X%gamma_beta.slice(r),1)));
    sigma2(r) = 1/rgamma(1,A_s2,1/B_s2)(0);
    
    if (Trace == 1) {
      if (size(find(printId == r+1))(0) == 1) Rcout << "Iteration " << r+1 << " of " << R << endl;
    }
  }
  
  beta.shed_slices(0,B-1);
  gamma.shed_slices(0,B-1);
  gamma_beta.shed_slices(0,B-1);
  omega.shed_slices(0,B-1);
  eta2.shed_rows(0,B-1);
  mu.shed_rows(0,B-1);
  m.shed_rows(0,B-1);
  sigma2.shed_rows(0,B-1);
  PIP_gamma.shed_slices(0,B-1);
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("gamma") = gamma,
    Rcpp::Named("PIP_gamma") = PIP_gamma,
    Rcpp::Named("gamma_beta") = gamma_beta,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("eta2") = eta2,
    Rcpp::Named("mu") = mu,
    Rcpp::Named("m") = m,
    Rcpp::Named("sigma2") = sigma2
  ); 
}


// [[Rcpp::export]]
Rcpp::List vloss(arma::vec y, arma::mat X, Rcpp::List hyper,
                  int R = 2000, int B = 1000, int Trace = 0) {
  
  double n = X.n_rows;
  double p = X.n_cols;
  
  arma::mat X2 = X%X;
  
  arma::vec beta_0 = inv_sympd(X.t()*X+0.05*eye(p,p))*X.t()*y;
  arma::mat B0 = zeros(n,p);
  for (int t = 0; t < n; t++) B0.row(t) = beta_0.t();
  
  double As = hyper["As"];
  double Bs = hyper["Bs"];
  
  double Ae = hyper["Ae"];
  double Be = hyper["Be"];
  
  arma::mat R_beta  = hyper["R1"];
  
  double tau2 = hyper["tau2"];
  double p0 = hyper["p0"];
  
  double xi2 = hyper["xi2"];
  arma::mat R_om  = hyper["ROm"];
  
  arma::mat Q_beta = inv_sympd(R_beta);
  double iQi = as_scalar(ones(n).t()*Q_beta*ones(n));
  
  arma::mat Q_om = inv_sympd(R_om);
  
  arma::vec sigma2 = ones(R);
  
  arma::cube beta = zeros(n,p,R);
  beta.slice(0) = B0;
  arma::cube gamma = ones(n,p,R);
  arma::cube PIP_gamma = ones(n,p,R);
  arma::cube gamma_beta = beta;
  arma::cube omega = zeros(n,p,R);
  
  arma::mat eta2 = ones(R,p);
  arma::cube z = 0.1*ones(n,p,R);
  
  arma::mat mu = zeros(R,p);
  mu.row(0) = beta_0.t();
  arma::mat m = zeros(R,p);
  
  arma::vec printId = regspace(1000, 1000, R);
  
  for (int r = 1; r < R; r++) {
    
    beta.slice(r) = beta.slice(r-1);
    gamma.slice(r) = gamma.slice(r-1);
    gamma_beta.slice(r) = gamma_beta.slice(r-1);
    omega.slice(r)  = omega.slice(r-1);
    z.slice(r)  = z.slice(r-1);
    
    sigma2(r) = sigma2(r-1);
    
    eta2.row(r) = eta2.row(r-1);
    mu.row(r) = mu.row(r-1);
    m.row(r) = m.row(r-1);
    
    arma::uvec Ord = randperm(p);
    
    for (int indJ = 0; indJ < p; indJ++) {
      double j = Ord(indJ);
      
      arma::vec res = y - sum(X%gamma.slice(r)%beta.slice(r),1) + X.col(j)%gamma.slice(r).col(j)%beta.slice(r).col(j);
      
      arma::mat Om_beta = 1/sigma2(r)*diagmat(gamma.slice(r).col(j)%X2.col(j)) + 1/eta2(r,j)*Q_beta;
      arma::mat Sigma_beta = inv_sympd(Om_beta);
      arma::vec mu_beta = Sigma_beta*(1/sigma2(r)*gamma.slice(r).col(j)%X.col(j)%res + 1/eta2(r,j)*Q_beta*ones(n)*mu(r,j));  
      beta.slice(r).col(j) = mvnrnd(mu_beta,Sigma_beta,1);
      
      double sigma2_mu = 1/(iQi/eta2(r,j) + 1/tau2);
      double mu_mu = sigma2_mu*as_scalar(1/eta2(r,j)*beta.slice(r).col(j).t()*Q_beta*ones(n));
      mu(r,j) = Rcpp::rnorm(1,mu_mu,sqrt(sigma2_mu))(0);
      
      double A_eta = Ae + 0.5*n;
      double B_eta = Be + 0.5*as_scalar((beta.slice(r).col(j)-mu(r,j)).t()*Q_beta*(beta.slice(r).col(j)-mu(r,j))); 
      eta2(r,j) = 1/rgamma(1,A_eta,1/B_eta)(0);
      
      
      arma::mat Om_om = diagmat(z.slice(r).col(j)) + 1/xi2*Q_om;
      arma::mat Sigma_om = inv_sympd(Om_om);
      arma::vec mu_om = Sigma_om*(gamma.slice(r).col(j)-0.5 + 1/xi2*Q_om*ones(n)*m(r,j));
      omega.slice(r).col(j) = mvnrnd(mu_om,Sigma_om,1);
      
      double sigma2_m = 1/(as_scalar(ones(n).t()*Q_om*ones(n))/xi2 + 1/tau2);
      double mu_m = sigma2_m*as_scalar(1/xi2*omega.slice(r).col(j).t()*Q_om*ones(n));
      m(r,j) = Rcpp::rnorm(1,mu_m,sqrt(sigma2_m))(0);
      
      z.slice(r).col(j) = rcpp_pgdraw(omega.slice(r).col(j));
      
      arma::vec omega_q = omega.slice(r).col(j)-0.5/sigma2(r)*(pow(beta.slice(r).col(j),2)%X2.col(j) - 2*beta.slice(r).col(j)%X.col(j)%res);
      omega_q(find(abs(omega_q)>50)) = sign(omega_q(find(abs(omega_q)>50)))*50;
      gamma.slice(r).col(j) = rbern(exp(omega_q)/(1+exp(omega_q)));
      PIP_gamma.slice(r).col(j) = exp(omega_q)/(1+exp(omega_q));
      
    }
    
    gamma_beta.slice(r) = gamma.slice(r)%beta.slice(r);
    
    double A_s2 = As + 0.5*n;
    double B_s2 = Bs + 0.5*sum((y-sum(X%gamma_beta.slice(r),1))%(y-sum(X%gamma_beta.slice(r),1)));
    sigma2(r) = 1/rgamma(1,A_s2,1/B_s2)(0);
    
    if (Trace == 1) {
      if (size(find(printId == r+1))(0) == 1) Rcout << "Iteration " << r+1 << " of " << R << endl;
    }
  }
  
  beta.shed_slices(0,B-1);
  gamma.shed_slices(0,B-1);
  gamma_beta.shed_slices(0,B-1);
  omega.shed_slices(0,B-1);
  eta2.shed_rows(0,B-1);
  mu.shed_rows(0,B-1);
  m.shed_rows(0,B-1);
  sigma2.shed_rows(0,B-1);
  PIP_gamma.shed_slices(0,B-1);
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("gamma") = gamma,
    Rcpp::Named("PIP_gamma") = PIP_gamma,
    Rcpp::Named("gamma_beta") = gamma_beta,
    Rcpp::Named("omega") = omega,
    Rcpp::Named("eta2") = eta2,
    Rcpp::Named("mu") = mu,
    Rcpp::Named("m") = m,
    Rcpp::Named("sigma2") = sigma2
  ); 
}

// [[Rcpp::export]]
Rcpp::List spgp(arma::vec y, arma::mat X, Rcpp::List hyper,
                int R = 2000, int B = 1000, int Trace = 0) {
  
  double n = X.n_rows;
  double p = X.n_cols;
  
  arma::mat X2 = X%X;
  
  arma::vec beta_0 = inv_sympd(X.t()*X+0.05*eye(p,p))*X.t()*y;
  arma::mat B0 = zeros(n,p);
  for (int t = 0; t < n; t++) B0.row(t) = beta_0.t();
  
  double As = hyper["As"];
  double Bs = hyper["Bs"];
  
  double Ae = hyper["Ae"];
  double Be = hyper["Be"];
  
  arma::mat R_beta  = hyper["R1"];
  
  double tau2 = hyper["tau2"];
  
  arma::mat Q_beta = inv_sympd(R_beta);
  double iQi = as_scalar(ones(n).t()*Q_beta*ones(n));
  
  arma::vec sigma2 = ones(R);
  
  arma::cube beta = zeros(n,p,R);
  beta.slice(0) = B0;
  
  arma::mat eta2 = ones(R,p);
  
  arma::mat mu = zeros(R,p);
  mu.row(0) = beta_0.t();
  
  arma::vec printId = regspace(1000, 1000, R);
  
  for (int r = 1; r < R; r++) {
    
    beta.slice(r) = beta.slice(r-1);
    sigma2(r) = sigma2(r-1);
    
    eta2.row(r) = eta2.row(r-1);
    mu.row(r) = mu.row(r-1);
    
    arma::uvec Ord = randperm(p);
    
    for (int indJ = 0; indJ < p; indJ++) {
      double j = Ord(indJ);
      
      arma::vec res = y - sum(X%beta.slice(r),1) + X.col(j)%beta.slice(r).col(j);
      
      arma::mat Om_beta = 1/sigma2(r)*diagmat(X2.col(j)) + 1/eta2(r,j)*Q_beta;
      arma::mat Sigma_beta = inv_sympd(Om_beta);
      arma::vec mu_beta = Sigma_beta*(1/sigma2(r)*X.col(j)%res + 1/eta2(r,j)*Q_beta*ones(n)*mu(r,j));  
      beta.slice(r).col(j) = mvnrnd(mu_beta,Sigma_beta,1);
      
      double sigma2_mu = 1/(iQi/eta2(r,j) + 1/tau2);
      double mu_mu = sigma2_mu*as_scalar(1/eta2(r,j)*beta.slice(r).col(j).t()*Q_beta*ones(n));
      mu(r,j) = Rcpp::rnorm(1,mu_mu,sqrt(sigma2_mu))(0);
      
      double A_eta = Ae + 0.5*n;
      double B_eta = Be + 0.5*as_scalar((beta.slice(r).col(j)-mu(r,j)).t()*Q_beta*(beta.slice(r).col(j)-mu(r,j))); 
      eta2(r,j) = 1/rgamma(1,A_eta,1/B_eta)(0);
      
    }
    
    double A_s2 = As + 0.5*n;
    double B_s2 = Bs + 0.5*sum((y-sum(X%beta.slice(r),1))%(y-sum(X%beta.slice(r),1)));
    sigma2(r) = 1/rgamma(1,A_s2,1/B_s2)(0);
    
    if (Trace == 1) {
      if (size(find(printId == r+1))(0) == 1) Rcout << "Iteration " << r+1 << " of " << R << endl;
    }
  }
  
  beta.shed_slices(0,B-1);
  eta2.shed_rows(0,B-1);
  mu.shed_rows(0,B-1);
  sigma2.shed_rows(0,B-1);
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("eta2") = eta2,
    Rcpp::Named("mu") = mu,
    Rcpp::Named("sigma2") = sigma2
  ); 
}
