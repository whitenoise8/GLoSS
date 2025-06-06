% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/wrapper_gloss.R
\name{GLoSS}
\alias{GLoSS}
\title{Inference for Global Local Spatial Selection with varying coefficients}
\usage{
GLoSS(
  y,
  X,
  hyper,
  fix_delta = NULL,
  local = TRUE,
  R = 2000,
  B = 1000,
  Trace = 0
)
}
\arguments{
\item{y}{vector of observations (response variable)}

\item{X}{design matrix (\code{n x p})}

\item{hyper}{list of hyperparameters (see paper for details)}

\item{fix_delta}{if 1 varying coefficients are considered (V-LoSS); if 0 constant coefficients are considered (C-LoSS); default is NULL, that is GLoSS is estimated}

\item{local}{if FALSE no local selection is performed; default is TRUE}

\item{R}{number of posterior draws}

\item{B}{number of burn-in draws. Final number of draws is \code{R-B}}

\item{Trace}{if 1, print progress of the algorithm}
}
\value{
Samples from posterior distributions according to the selected method (see the paper for details, the names of the variables match)
}
\description{
Inference for Global Local Spatial Selection with varying coefficients
}
