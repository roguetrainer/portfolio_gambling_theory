"""
Single-Period Portfolio Optimization

Implements various single-period portfolio selection criteria:
- Mean-Variance (Markowitz)
- Expected Utility (CRRA, CARA, Log)
- VaR/CVaR
- Safety-First
"""

import numpy as np
from scipy.optimize import minimize


class SinglePeriodOptimizer:
    """Single-period portfolio optimization with multiple criteria."""
    
    def __init__(self, returns, cov_matrix=None):
        """
        Initialize optimizer with return data.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns matrix (n_periods x n_assets)
        cov_matrix : np.ndarray, optional
            Covariance matrix. If None, computed from returns.
        """
        self.returns = np.array(returns)
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = cov_matrix if cov_matrix is not None else np.cov(returns.T)
        self.n_assets = len(self.mean_returns)
        
    def mean_variance(self, target_return=None, risk_aversion=None):
        """
        Mean-variance optimization (Markowitz).
        
        Can either:
        1. Minimize variance for target return
        2. Maximize return - (lambda/2)*variance
        
        Parameters:
        -----------
        target_return : float, optional
            Target expected return
        risk_aversion : float, optional
            Risk aversion parameter lambda
            
        Returns:
        --------
        weights : np.ndarray
            Optimal portfolio weights
        """
        if target_return is not None:
            return self._minimize_variance_target_return(target_return)
        elif risk_aversion is not None:
            return self._maximize_utility_quadratic(risk_aversion)
        else:
            # Default: tangency portfolio (max Sharpe ratio)
            return self._tangency_portfolio()
    
    def _minimize_variance_target_return(self, target_return):
        """Minimize variance subject to target return."""
        def objective(w):
            return w @ self.cov_matrix @ w
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w @ self.mean_returns - target_return}
        ]
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(objective, 
                         x0=np.ones(self.n_assets)/self.n_assets,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        return result.x
    
    def _maximize_utility_quadratic(self, risk_aversion):
        """Maximize E[R] - (lambda/2)*Var[R]."""
        def objective(w):
            expected_return = w @ self.mean_returns
            variance = w @ self.cov_matrix @ w
            return -(expected_return - 0.5 * risk_aversion * variance)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(objective,
                         x0=np.ones(self.n_assets)/self.n_assets,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        return result.x
    
    def _tangency_portfolio(self):
        """Maximum Sharpe ratio portfolio (assuming rf=0)."""
        inv_cov = np.linalg.inv(self.cov_matrix)
        w = inv_cov @ self.mean_returns
        return w / np.sum(w)
    
    def expected_utility(self, utility_type='log', gamma=2.0, alpha=1.0):
        """
        Expected utility maximization.
        
        Parameters:
        -----------
        utility_type : str
            'log', 'power' (CRRA), or 'exponential' (CARA)
        gamma : float
            Relative risk aversion for CRRA (power utility)
        alpha : float
            Absolute risk aversion for CARA (exponential utility)
            
        Returns:
        --------
        weights : np.ndarray
            Optimal portfolio weights
        """
        def utility(wealth):
            if utility_type == 'log':
                return np.log(np.maximum(wealth, 1e-10))
            elif utility_type == 'power':
                if gamma == 1:
                    return np.log(np.maximum(wealth, 1e-10))
                return (wealth**(1-gamma) - 1) / (1 - gamma)
            elif utility_type == 'exponential':
                return -np.exp(-alpha * wealth)
            else:
                raise ValueError(f"Unknown utility type: {utility_type}")
        
        def objective(w):
            portfolio_returns = self.returns @ w
            terminal_wealth = 1 + portfolio_returns
            expected_util = np.mean(utility(terminal_wealth))
            return -expected_util
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(objective,
                         x0=np.ones(self.n_assets)/self.n_assets,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        return result.x
    
    def var_cvar(self, alpha=0.05, method='cvar'):
        """
        VaR or CVaR optimization.
        
        Parameters:
        -----------
        alpha : float
            Confidence level (e.g., 0.05 for 95% VaR)
        method : str
            'var' or 'cvar'
            
        Returns:
        --------
        weights : np.ndarray
            Optimal portfolio weights
        """
        def portfolio_var(w, alpha):
            portfolio_returns = self.returns @ w
            return -np.percentile(portfolio_returns, alpha * 100)
        
        def portfolio_cvar(w, alpha):
            portfolio_returns = self.returns @ w
            var = -np.percentile(portfolio_returns, alpha * 100)
            return -np.mean(portfolio_returns[portfolio_returns <= -var])
        
        if method == 'var':
            objective = lambda w: portfolio_var(w, alpha)
        else:
            objective = lambda w: portfolio_cvar(w, alpha)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(objective,
                         x0=np.ones(self.n_assets)/self.n_assets,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        return result.x
    
    def safety_first(self, disaster_level=-0.10):
        """
        Roy's safety-first criterion: maximize (μ - d) / σ
        
        Parameters:
        -----------
        disaster_level : float
            Minimum acceptable return
            
        Returns:
        --------
        weights : np.ndarray
            Optimal portfolio weights
        """
        def objective(w):
            expected_return = w @ self.mean_returns
            std_dev = np.sqrt(w @ self.cov_matrix @ w)
            if std_dev < 1e-10:
                return 1e10
            return -(expected_return - disaster_level) / std_dev
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(objective,
                         x0=np.ones(self.n_assets)/self.n_assets,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        return result.x
    
    def efficient_frontier(self, n_points=50):
        """Compute the efficient frontier."""
        min_return = np.min(self.mean_returns)
        max_return = np.max(self.mean_returns)
        target_returns = np.linspace(min_return, max_return, n_points)
        
        volatilities = []
        weights_list = []
        
        for target_ret in target_returns:
            try:
                w = self._minimize_variance_target_return(target_ret)
                vol = np.sqrt(w @ self.cov_matrix @ w)
                volatilities.append(vol)
                weights_list.append(w)
            except:
                continue
        
        return {
            'returns': target_returns[:len(volatilities)],
            'volatilities': np.array(volatilities),
            'weights': np.array(weights_list)
        }
    
    def portfolio_metrics(self, weights):
        """Calculate portfolio metrics for given weights."""
        expected_return = weights @ self.mean_returns
        volatility = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        portfolio_returns = self.returns @ weights
        var_95 = -np.percentile(portfolio_returns, 5)
        cvar_95 = -np.mean(portfolio_returns[portfolio_returns <= -var_95])
        
        return {
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': self._max_drawdown(portfolio_returns),
            'weights': weights
        }
    
    def _max_drawdown(self, returns):
        """Calculate maximum drawdown from return series."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
