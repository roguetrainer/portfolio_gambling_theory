# Implementation Guide

## Package Structure

```
portfolio_gambling_theory/
├── src/portfolio_gambling/
│   ├── __init__.py              # Package initialization
│   ├── single_period.py         # Single-period optimization
│   ├── multi_period.py          # Multi-period and Kelly
│   ├── gambling_strategies.py   # Bold/timid play
│   ├── market_simulation.py     # Data generation
│   ├── visualization.py         # Plotting utilities
│   └── utils.py                 # Helper functions
├── notebooks/                    # Jupyter notebooks
├── docs/                         # Documentation
├── tests/                        # Unit tests
└── examples/                     # Example scripts
```

## Installation

```bash
# Clone/download the package
cd portfolio_gambling_theory

# Install in development mode
pip install -e .

# Or install requirements only
pip install -r requirements.txt
```

## Core Modules

### 1. single_period.py

Implements single-period portfolio optimization criteria.

**Classes**:
- `MeanVarianceOptimizer`: Markowitz mean-variance optimization
- `ExpectedUtilityOptimizer`: Expected utility maximization (CRRA, CARA, log)
- `VaROptimizer`: VaR and CVaR minimization
- `SafetyFirstOptimizer`: Roy's safety-first criterion

**Example Usage**:

```python
from portfolio_gambling import MeanVarianceOptimizer, MarketSimulator

# Generate data
sim = MarketSimulator(n_assets=5, n_periods=252)
returns = sim.generate_returns()

# Optimize portfolio
optimizer = MeanVarianceOptimizer(risk_aversion=2.0, allow_short=False)
weights = optimizer.optimize(returns)

print(f"Optimal weights: {weights}")
print(f"Sum of weights: {weights.sum()}")
```

### 2. multi_period.py

Implements multi-period and infinite-horizon strategies.

**Classes**:
- `KellyOptimizer`: Kelly criterion for growth-optimal portfolios
- `MertonPortfolio`: Merton's continuous-time portfolio selection
- `DynamicProgrammingOptimizer`: Finite-horizon dynamic programming
- `ProportionalRebalancing`: Constant-weight rebalancing strategy

**Example Usage**:

```python
from portfolio_gambling import KellyOptimizer

# Kelly optimization
kelly = KellyOptimizer(fraction=1.0, allow_short=False)
kelly_weights = kelly.optimize(returns)

# Compute growth rate
growth_rate = kelly.growth_rate(returns, kelly_weights)
print(f"Geometric growth rate: {growth_rate:.2%}")

# Fractional Kelly (more conservative)
half_kelly = KellyOptimizer(fraction=0.5, allow_short=False)
half_kelly_weights = half_kelly.optimize(returns)
```

### 3. gambling_strategies.py

Implements gambling strategies from Dubins & Savage.

**Classes**:
- `BoldPlayStrategy`: Optimal strategy for subfair games
- `TimidPlayStrategy`: Minimal betting strategy
- `ProportionalBetting`: Fixed-fraction betting (includes Kelly)

**Example Usage**:

```python
from portfolio_gambling import BoldPlayStrategy, ProportionalBetting

# Bold play for subfair game
bold = BoldPlayStrategy(
    target_wealth=2.0,
    win_prob=0.49,  # Subfair
    odds=1.0
)

# Simulate
success_rate, paths = bold.simulate(
    initial_wealth=1.0,
    n_simulations=10000
)

print(f"Empirical success rate: {success_rate:.3f}")
print(f"Theoretical prob: {bold.success_probability(1.0):.3f}")

# Kelly betting for favorable game
kelly_frac = ProportionalBetting.kelly_fraction(win_prob=0.55, odds=1.0)
prop = ProportionalBetting(kelly_frac, win_prob=0.55)

paths, mean_wealth, median_wealth = prop.simulate(
    initial_wealth=1.0,
    n_rounds=1000,
    n_simulations=1000
)
```

### 4. market_simulation.py

Generate synthetic market data for testing.

**Classes**:
- `MarketSimulator`: Basic correlated returns
- `GeometricBrownianMotion`: GBM price simulation
- `RegimeSwitchingMarket`: Bull/normal/bear regimes
- `FactorModel`: Factor-based returns

**Example Usage**:

```python
from portfolio_gambling import MarketSimulator, GeometricBrownianMotion

# Basic simulation
sim = MarketSimulator(
    n_assets=5,
    n_periods=252,
    mean_returns=np.array([0.08, 0.10, 0.12, 0.15, 0.18]),
    volatilities=np.array([0.15, 0.18, 0.22, 0.28, 0.35]),
    correlation=0.3,
    random_seed=42
)
returns = sim.generate_returns(frequency='daily')

# Geometric Brownian Motion
gbm = GeometricBrownianMotion(
    n_assets=3,
    n_periods=252,
    mu=np.array([0.10, 0.12, 0.15]),
    sigma=np.array([0.20, 0.25, 0.30]),
    initial_prices=np.array([100, 100, 100])
)
prices, returns = gbm.simulate_prices()
```

### 5. visualization.py

Plotting utilities for portfolio analysis.

**Functions**:
- `plot_efficient_frontier()`: Plot mean-variance efficient frontier
- `plot_wealth_paths()`: Plot wealth evolution over time
- `plot_strategy_comparison()`: Compare multiple strategies
- `plot_portfolio_weights()`: Visualize portfolio allocations
- `plot_return_distribution()`: Compare return distributions
- `plot_growth_comparison()`: Compare long-term growth

**Example Usage**:

```python
from portfolio_gambling.visualization import (
    plot_efficient_frontier,
    plot_portfolio_weights,
    plot_wealth_paths
)

# Plot efficient frontier
mv = MeanVarianceOptimizer()
frontier_returns, frontier_stds = mv.efficient_frontier(returns)

fig = plot_efficient_frontier(
    frontier_returns,
    frontier_stds,
    portfolios={
        'Kelly': (kelly_return, kelly_std),
        'Min Variance': (mv_return, mv_std)
    },
    title="Efficient Frontier Comparison"
)

# Plot weights
fig = plot_portfolio_weights(
    {'Kelly': kelly_weights, 'Mean-Variance': mv_weights},
    asset_names=['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'],
    title="Portfolio Allocations"
)

# Plot wealth paths
fig = plot_wealth_paths(
    {'Bold Play': bold_paths, 'Timid Play': timid_paths},
    target_wealth=2.0,
    title="Bold vs. Timid Play"
)
```

### 6. utils.py

Helper functions for portfolio analysis.

**Functions**:
- `calculate_portfolio_metrics()`: Compute comprehensive metrics
- `calculate_max_drawdown()`: Maximum drawdown
- `generate_correlation_matrix()`: Create correlation matrices
- `annualize_metrics()`: Convert to annual terms
- `calculate_turnover()`: Portfolio turnover
- `bootstrap_returns()`: Bootstrap for robustness testing
- `format_portfolio_summary()`: Pretty-print portfolio info

**Example Usage**:

```python
from portfolio_gambling.utils import (
    calculate_portfolio_metrics,
    format_portfolio_summary,
    annualize_metrics
)

# Calculate metrics
metrics = calculate_portfolio_metrics(
    returns,
    weights,
    risk_free_rate=0.02/252  # Daily risk-free rate
)

# Format summary
summary = format_portfolio_summary(
    weights,
    metrics,
    asset_names=['Asset 1', 'Asset 2', 'Asset 3']
)
print(summary)

# Annualize
annual_return, annual_vol = annualize_metrics(
    metrics['mean_return'],
    metrics['std_dev'],
    periods_per_year=252
)
```

## Complete Example Workflow

```python
import numpy as np
from portfolio_gambling import (
    MeanVarianceOptimizer,
    KellyOptimizer,
    BoldPlayStrategy,
    MarketSimulator,
)
from portfolio_gambling.visualization import plot_strategy_comparison
from portfolio_gambling.utils import calculate_portfolio_metrics

# 1. Generate market data
sim = MarketSimulator(n_assets=5, n_periods=252, random_seed=42)
returns = sim.generate_returns()

# 2. Optimize portfolios
mv = MeanVarianceOptimizer(risk_aversion=2.0)
kelly = KellyOptimizer(fraction=1.0)

mv_weights = mv.optimize(returns)
kelly_weights = kelly.optimize(returns)

# 3. Calculate metrics
mv_metrics = calculate_portfolio_metrics(returns, mv_weights)
kelly_metrics = calculate_portfolio_metrics(returns, kelly_weights)

# 4. Compare
results = {
    'Mean-Variance': mv_metrics,
    'Kelly': kelly_metrics
}

fig = plot_strategy_comparison(
    results,
    metrics=['sharpe_ratio', 'max_drawdown'],
    title="Strategy Comparison"
)

# 5. Simulate gambling strategies
bold = BoldPlayStrategy(target_wealth=2.0, win_prob=0.49)
success_rate, paths = bold.simulate(initial_wealth=1.0, n_simulations=10000)

print(f"Bold play success rate: {success_rate:.2%}")
```

## Testing

```python
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_single_period.py

# Run with coverage
pytest --cov=portfolio_gambling tests/
```

## Jupyter Notebooks

The `notebooks/` directory contains interactive tutorials:

1. `01_single_period_comparison.ipynb`: Compare single-period criteria
2. `02_kelly_criterion.ipynb`: Deep dive into Kelly betting
3. `03_bold_vs_timid_play.ipynb`: Dubins-Savage gambling strategies
4. `04_merton_portfolio.ipynb`: Continuous-time portfolio selection
5. `05_transaction_costs.ipynb`: Impact of transaction costs
6. `06_robust_optimization.ipynb`: Handling parameter uncertainty
7. `07_comprehensive_comparison.ipynb`: All strategies on realistic data

## Advanced Topics

### Custom Utility Functions

```python
# Define custom utility
def custom_utility(wealth, gamma=2.0, lambda_loss=2.5):
    """Prospect theory-like utility"""
    if wealth >= 1:
        return (wealth - 1) ** (1 - gamma)
    else:
        return -lambda_loss * (1 - wealth) ** (1 - gamma)

# Use with optimizer (requires modification)
from scipy.optimize import minimize

def objective(w):
    portfolio_returns = returns @ w
    terminal_wealth = 1 + portfolio_returns
    return -np.mean([custom_utility(w) for w in terminal_wealth])
```

### Parameter Estimation

```python
# Robust covariance estimation
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
cov_robust = lw.fit(returns).covariance_

# Use robust estimates
optimizer = MeanVarianceOptimizer(risk_aversion=2.0)
# Manually set covariance in optimization
```

### Backtesting

```python
def backtest_strategy(optimizer, returns, rebalance_freq=20):
    """Simple backtesting framework"""
    n_periods = len(returns)
    wealth = np.ones(n_periods + 1)
    
    for t in range(0, n_periods, rebalance_freq):
        # Use past data to optimize
        past_returns = returns[max(0, t-252):t]
        if len(past_returns) < 50:
            continue
            
        weights = optimizer.optimize(past_returns)
        
        # Apply weights for next rebalance period
        for s in range(t, min(t + rebalance_freq, n_periods)):
            ret = returns[s] @ weights
            wealth[s + 1] = wealth[s] * (1 + ret)
    
    return wealth
```

## Troubleshooting

### Common Issues

1. **Optimization fails to converge**:
   - Check if covariance matrix is positive definite
   - Try different initial guess
   - Relax constraints slightly

2. **Kelly weights sum to > 1 (leverage)**:
   - Add leverage constraint: `max_leverage=1.0`
   - Use fractional Kelly: `fraction=0.5`

3. **Negative weights despite no-short constraint**:
   - Numerical precision issue
   - Clip: `weights = np.maximum(weights, 0)`

4. **Poor out-of-sample performance**:
   - Estimation error in μ, Σ
   - Use robust estimation or shrinkage
   - Reduce turnover with transaction costs

## Performance Optimization

```python
# Use JAX for faster optimization
import jax
import jax.numpy as jnp

@jax.jit
def kelly_objective(w, returns):
    portfolio_returns = jnp.dot(returns, w)
    return -jnp.mean(jnp.log(1 + portfolio_returns))

# Parallel simulations
from joblib import Parallel, delayed

def run_simulation(seed):
    np.random.seed(seed)
    # ... simulation code ...
    return result

results = Parallel(n_jobs=-1)(
    delayed(run_simulation)(seed) for seed in range(1000)
)
```

## References and Further Reading

- Documentation: `/docs/theoretical_foundations.md`
- Original papers in bibliography
- Example notebooks for hands-on learning
- Test suite for usage examples
