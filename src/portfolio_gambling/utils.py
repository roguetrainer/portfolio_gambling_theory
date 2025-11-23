"""
Utility functions for portfolio gambling theory.
"""

import numpy as np


def generate_returns(n_assets=5, n_periods=1000, mean_returns=None, 
                    cov_matrix=None, return_type='normal'):
    """
    Generate synthetic return data.
    
    Parameters:
    -----------
    n_assets : int
        Number of assets
    n_periods : int
        Number of time periods
    mean_returns : np.ndarray, optional
        Mean returns for each asset
    cov_matrix : np.ndarray, optional
        Covariance matrix
    return_type : str
        'normal' or 't' distribution
        
    Returns:
    --------
    returns : np.ndarray
        Generated returns (n_periods x n_assets)
    """
    if mean_returns is None:
        mean_returns = np.random.uniform(0.05, 0.15, n_assets) / 252
    
    if cov_matrix is None:
        # Generate random covariance matrix
        A = np.random.randn(n_assets, n_assets)
        cov_matrix = (A @ A.T) / (252 * 100)  # Daily volatility ~15-25%
    
    if return_type == 'normal':
        returns = np.random.multivariate_normal(mean_returns, cov_matrix, n_periods)
    elif return_type == 't':
        # Student t with df=5 for fat tails
        from scipy.stats import multivariate_t
        returns = multivariate_t.rvs(mean_returns, cov_matrix, df=5, size=n_periods)
    else:
        raise ValueError(f"Unknown return_type: {return_type}")
    
    return returns


def sharpe_ratio(returns, weights):
    """Calculate Sharpe ratio for a portfolio."""
    port_returns = returns @ weights
    return np.mean(port_returns) / np.std(port_returns) if np.std(port_returns) > 0 else 0


def portfolio_return(returns, weights):
    """Calculate portfolio return."""
    return returns @ weights


def portfolio_volatility(cov_matrix, weights):
    """Calculate portfolio volatility."""
    return np.sqrt(weights @ cov_matrix @ weights)
