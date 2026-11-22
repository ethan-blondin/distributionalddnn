from scipy.optimize import minimize
import numpy as np
from numpy.linalg import lstsq as ldiv

def quantreg(x, y, tau):
    """
    outputs a 1D NumPy array of optimal regression weights

    x: Design matrix of shape (182, K) containing a constant intercept column plus 
    the historical point forecasts over the 182-day calibration window, either the 
    4 separate point forecasts (LEAR-56/84/1092/1456 or DNN-1..4) for QRA, or the 
    single averaged point forecast (LEAR-Ens or DNN-Ens) for QRM.

    y: Vector of length 182 containing the actual realized electricity prices for 
    a single target hour h over those same 182 days.

    tau: Target quantile level (e.g., 0.05 for the 5th percentile, 0.50 for the median).
    """
    def rho(r):
        """
        computes the pinball loss function with first the residual and then the asymetric loss.
        """
        temp = y - np.dot(x,r) # computes the residuals
        return np.sum(np.abs(temp - (temp < 0) * temp / tau))
    pmean = ldiv(x, y, rcond=None)[0] # outputs a 1D NumPy array of initial regression coefficients, one weight per column in x—computed via OLS
    # return minimize(rho, pmean, method='SLSQP', options={'disp': False}).x
    # return minimize(rho, pmean, method='SLSQP', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='COBYLA', options={'disp': False}).x
    # return minimize(rho, pmean, method='COBYLA', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='TNC', options={'disp': False}).x
    # return minimize(rho, pmean, method='TNC', tol=None, options={'maxiter': 25, 'disp': False}).x
    return minimize(rho, pmean, method='TNC', tol=None, options={'maxiter': 10 * len(pmean), 'disp': False}).x
    # return minimize(rho, pmean, method='L-BFGS-B', options={'disp': False}).x
    # return minimize(rho, pmean, method='L-BFGS-B', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='BFGS', options={'disp': False}).x
    # return minimize(rho, pmean, method='BFGS', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='CG', options={'disp': False}).x
    # return minimize(rho, pmean, method='CG', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='Powell', options={'disp': False}).x
    # return minimize(rho, pmean, method='Powell', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='Nelder-Mead', tol=None, options={'maxiter': 25, 'disp': False}).x
    # return minimize(rho, pmean, method='Nelder-Mead', options={'disp': False}).x
