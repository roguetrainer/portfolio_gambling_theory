"""
Multi-Period Portfolio Optimization

Implements multi-period and infinite-horizon strategies:
- Kelly Criterion (growth-optimal)
- Fractional Kelly
- Dynamic rebalancing strategies
- Merton portfolio (continuous-time approximation)
"""

import numpy as np
from scipy.optimize import minimize


class MultiPeriodOptimizer:
    """Multi-period portfolio optimization strategies."""
    
    def __init__(self, returns):
        """
        Initialize with historical returns.
        
        Parameters:
        -----------
        returns : np.ndarray
            Historical returns matrix (n_periods x n_assets)
        """
        self.returns = np.array(returns)
        self.mean_returns = np.mean(returns, axis=0)
        self.cov_matrix = np.cov(returns.T)
        self.n_assets = len(self.mean_returns)
    
    def kelly_criterion(self):
        """
        Kelly criterion: maximize E[log(1 + r'w)].
        
        This is the growth-optimal portfolio that maximizes
        the expected logarithmic growth rate of wealth.
        
        Returns:
        --------
        weights : np.ndarray
            Optimal Kelly weights
        """
        def objective(w):
            portfolio_returns = self.returns @ w
            log_returns = np.log(1 + portfolio_returns)
            return -np.mean(log_returns)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(self.n_assets)]
        
        result = minimize(objective,
                         x0=np.ones(self.n_assets)/self.n_assets,
                         method='SLSQP',
                         bounds=bounds,
                         constraints=constraints)
        return result.x
    
    def fractional_kelly(self, fraction=0.5):
        """
        Fractional Kelly: bet fraction * kelly_weights.
        
        This reduces volatility at the cost of some growth.
        
        Parameters:
        -----------
        fraction : float
            Fraction of Kelly bet (0 < fraction <= 1)
            
        Returns:
        --------
        weights : np.ndarray
            Fractional Kelly weights
        """
        kelly_w = self.kelly_criterion()
        frac_kelly_w = fraction * kelly_w
        # Remaining goes to cash (assumed to be zero return)
        return frac_kelly_w
    
    def merton_portfolio(self):
        """
        Merton's continuous-time portfolio.
        
        For log utility, this is identical to Kelly criterion.
        Analytical solution: w = Σ^(-1) * μ
        
        Returns:
        --------
        weights : np.ndarray
            Merton portfolio weights
        """
        try:
            inv_cov = np.linalg.inv(self.cov_matrix)
            w = inv_cov @ self.mean_returns
            # Normalize to sum to 1
            return w / np.sum(w)
        except np.linalg.LinAlgError:
            # Fallback to Kelly if covariance is singular
            return self.kelly_criterion()
    
    def simulate_strategy(self, strategy_weights, n_periods=252, n_simulations=1000, 
                         rebalance_freq=1):
        """
        Simulate a strategy over time with rebalancing.
        
        Parameters:
        -----------
        strategy_weights : np.ndarray or str
            Portfolio weights or strategy name ('kelly', 'equal', 'mean_variance')
        n_periods : int
            Number of periods to simulate
        n_simulations : int
            Number of simulation paths
        rebalance_freq : int
            Rebalance every N periods
            
        Returns:
        --------
        results : dict
            Simulation results with wealth paths and statistics
        """
        if isinstance(strategy_weights, str):
            if strategy_weights == 'kelly':
                weights = self.kelly_criterion()
            elif strategy_weights == 'equal':
                weights = np.ones(self.n_assets) / self.n_assets
            else:
                raise ValueError(f"Unknown strategy: {strategy_weights}")
        else:
            weights = strategy_weights
        
        wealth_paths = np.zeros((n_simulations, n_periods + 1))
        wealth_paths[:, 0] = 1.0
        
        for sim in range(n_simulations):
            for t in range(n_periods):
                # Sample returns from historical distribution
                idx = np.random.randint(0, len(self.returns))
                period_returns = self.returns[idx]
                
                # Portfolio return
                port_return = weights @ period_returns
                wealth_paths[sim, t+1] = wealth_paths[sim, t] * (1 + port_return)
                
                # Rebalance if needed (geometric drift means weights stay constant)
                # In practice, rebalancing maintains constant weights
        
        # Calculate statistics
        final_wealth = wealth_paths[:, -1]
        terminal_returns = (final_wealth - 1) / 1
        
        return {
            'wealth_paths': wealth_paths,
            'mean_final_wealth': np.mean(final_wealth),
            'median_final_wealth': np.median(final_wealth),
            'std_final_wealth': np.std(final_wealth),
            'prob_profit': np.mean(final_wealth > 1),
            'geometric_mean_return': np.exp(np.mean(np.log(final_wealth))) - 1,
            'cagr': (np.mean(final_wealth)) ** (1/n_periods) - 1,
            'sharpe_ratio': np.mean(terminal_returns) / np.std(terminal_returns) if np.std(terminal_returns) > 0 else 0
        }
    
    def compare_strategies(self, strategies=None, n_periods=252, n_simulations=1000):
        """
        Compare multiple strategies via simulation.
        
        Parameters:
        -----------
        strategies : list of tuples
            List of (name, weights) tuples
        n_periods : int
            Simulation length
        n_simulations : int
            Number of simulations
            
        Returns:
        --------
        comparison : dict
            Results for each strategy
        """
        if strategies is None:
            kelly_w = self.kelly_criterion()
            half_kelly_w = self.fractional_kelly(0.5)
            equal_w = np.ones(self.n_assets) / self.n_assets
            
            strategies = [
                ('Kelly (Full)', kelly_w),
                ('Half Kelly', half_kelly_w),
                ('Equal Weight', equal_w),
            ]
        
        results = {}
        for name, weights in strategies:
            results[name] = self.simulate_strategy(weights, n_periods, n_simulations)
        
        return results
    
    def dynamic_kelly(self, returns_sequence):
        """
        Dynamic Kelly rebalancing over a sequence of returns.
        
        Parameters:
        -----------
        returns_sequence : np.ndarray
            Sequence of return realizations (n_periods x n_assets)
            
        Returns:
        --------
        results : dict
            Wealth path and rebalancing history
        """
        n_periods = len(returns_sequence)
        wealth = np.zeros(n_periods + 1)
        wealth[0] = 1.0
        weights_history = []
        
        for t in range(n_periods):
            # Compute Kelly weights based on historical data up to time t
            if t < 20:  # Need minimum history
                w = np.ones(self.n_assets) / self.n_assets
            else:
                hist_returns = self.returns[:t] if t < len(self.returns) else self.returns
                temp_opt = MultiPeriodOptimizer(hist_returns)
                w = temp_opt.kelly_criterion()
            
            weights_history.append(w)
            
            # Apply return
            port_return = w @ returns_sequence[t]
            wealth[t+1] = wealth[t] * (1 + port_return)
        
        return {
            'wealth': wealth,
            'weights_history': np.array(weights_history),
            'final_wealth': wealth[-1],
            'cagr': (wealth[-1]) ** (1/n_periods) - 1
        }
    
    def time_series_growth_rate(self, weights):
        """
        Calculate realized growth rate for given weights.
        
        Parameters:
        -----------
        weights : np.ndarray
            Portfolio weights
            
        Returns:
        --------
        growth_rate : float
            Geometric mean growth rate
        """
        portfolio_returns = self.returns @ weights
        log_returns = np.log(1 + portfolio_returns)
        return np.mean(log_returns)
    
    def kelly_leverage(self):
        """
        Calculate Kelly leverage ratio.
        
        Returns the ratio of Kelly portfolio to max Sharpe portfolio,
        showing how much leverage Kelly implies.
        
        Returns:
        --------
        leverage : float
            Kelly leverage ratio
        """
        kelly_w = self.kelly_criterion()
        
        # Max Sharpe (tangency) portfolio
        inv_cov = np.linalg.inv(self.cov_matrix)
        sharpe_w = inv_cov @ self.mean_returns
        sharpe_w = sharpe_w / np.sum(sharpe_w)
        
        # Leverage is ratio of sum of absolute weights
        kelly_leverage = np.sum(np.abs(kelly_w))
        sharpe_leverage = np.sum(np.abs(sharpe_w))
        
        return kelly_leverage / sharpe_leverage if sharpe_leverage > 0 else 1.0
