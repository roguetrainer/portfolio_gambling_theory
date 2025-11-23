# Portfolio Gambling Theory Package - Complete Summary

## Package Contents

This is a comprehensive Python package implementing portfolio optimization and gambling theory concepts from classic papers including Markowitz (1952), Kelly (1956), Dubins & Savage (1965), and Merton (1969).

## Core Implementations

### 1. Single-Period Optimization (`single_period.py`)
- **Mean-Variance optimization**: Minimize variance for target return, maximize Sharpe ratio
- **Expected Utility maximization**: Log, CRRA (power), CARA (exponential) utilities
- **VaR/CVaR optimization**: Tail risk management
- **Safety-First criterion**: Roy's disaster avoidance approach
- **Efficient frontier** computation
- Complete portfolio metrics (Sharpe ratio, drawdowns, etc.)

### 2. Multi-Period Strategies (`multi_period.py`)
- **Kelly Criterion**: Growth-optimal portfolio maximizing E[log(wealth)]
- **Fractional Kelly**: Conservative variants (e.g., half-Kelly)
- **Merton portfolio**: Continuous-time approximation
- **Monte Carlo simulations**: Multi-period wealth path simulations
- **Dynamic rebalancing**: Time-varying optimal strategies
- **Kelly leverage** analysis
- Strategy comparison framework

### 3. Gambling Strategies (`gambling_strategies.py`)
- **Bold Play**: Optimal strategy for subfair games (bet max possible)
- **Timid Play**: Small bet strategy for comparison
- **Probability calculations**: Analytical and recursive methods for reaching targets
- **Monte Carlo gambling simulations**: Empirical verification
- **Strategy comparisons**: Bold vs timid with different game parameters
- **Superfair vs subfair** game analysis

### 4. Utilities (`utils.py`)
- Synthetic market data generation (normal and t-distributions)
- Portfolio metrics calculations
- Sharpe ratio, volatility, returns

## Jupyter Notebooks (4 comprehensive notebooks)

### Notebook 1: Mean-Variance vs Kelly Criterion
**Key Topics:**
- Comparison of single-period (Markowitz) vs infinite-horizon (Kelly) optimization
- Efficient frontier analysis
- Multi-period simulations showing mean vs geometric mean
- Kelly leverage analysis
- Visualization of wealth paths and distributions
- Theoretical connections: Kelly ≈ log utility ≈ Mean-variance with λ=2

**Insights:**
- Kelly maximizes geometric mean (median wealth), not arithmetic mean
- Kelly often lies below efficient frontier in mean-std space
- Fractional Kelly reduces volatility
- Long-run: Kelly is asymptotically optimal

### Notebook 2: Dubins-Savage Gambling Strategies  
**Key Topics:**
- Implementation of "How to Gamble If You Must" (1965)
- Bold play vs timid play for subfair games (e.g., roulette)
- Theoretical probability calculations
- Monte Carlo verification of strategies
- Effect of initial wealth on success probability
- Superfair game analysis (when you have an edge)
- "Double your money" problem
- Connections to Kelly criterion

**Insights:**
- For subfair games: bold play is mathematically optimal
- Bold play minimizes exposure to house edge
- Binary outcomes: ruin or success
- Different from Kelly: finite-horizon target vs infinite-horizon growth
- Practical wisdom: "If you must gamble (subfair), go big or go home!"

### Notebook 3: Utility Maximization
**Key Topics:**
- Logarithmic utility (equivalent to Kelly)
- Power utility (CRRA) with different risk aversions (γ)
- Exponential utility (CARA) 
- Visualization of utility functions and marginal utilities
- Portfolio optimization under different utilities
- Risk aversion spectrum analysis (γ from 0.1 to 10)
- Certainty equivalent calculations
- Comparison with mean-variance

**Insights:**
- Higher γ (risk aversion) → safer asset allocation
- Log utility (γ=1) = Kelly criterion
- Different utilities → different optimal portfolios
- Certainty equivalent useful for comparing portfolios
- Beyond mean-variance: captures full distribution

### Notebook 5: Multi-Period Rebalancing
**Key Topics:**
- Dynamic strategies across regime changes (bull, bear, sideways)
- Rebalancing frequency comparison (daily, weekly, monthly, quarterly, annual, buy-hold)
- Transaction cost impact analysis (0 to 50 basis points)
- Strategy comparison (Kelly, half-Kelly, mean-variance, equal weight, risk parity)
- Static vs dynamic allocation
- Weight drift and turnover analysis
- Drawdown analysis

**Insights:**
- Sweet spot: monthly to quarterly rebalancing
- Transaction costs matter significantly
- Higher costs → less frequent rebalancing optimal
- Dynamic Kelly adapts to regimes but higher turnover
- Practical: quarterly rebalancing for retail, monthly with thresholds for institutional

## Mathematical Rigor

All implementations include:
- Proper optimization with scipy.optimize
- Numerical stability considerations
- Edge case handling
- Monte Carlo verification where appropriate
- Analytical solutions where available

## Visualization

Each notebook includes:
- Wealth path evolution plots
- Distribution histograms
- Efficient frontiers
- Weight allocation charts
- Risk-return scatter plots
- Comparative analysis visualizations
- Regime change marking

## Practical Applications

**Portfolio Management:**
- Kelly for long-term growth maximization
- Mean-variance for single-period optimization
- Fractional Kelly for risk management
- Dynamic rebalancing with transaction costs

**Risk Management:**
- VaR/CVaR for tail risk
- Drawdown analysis
- Safety-first for disaster avoidance

**Gambling/Trading:**
- Bold play when forced into negative expectation
- Kelly when you have an edge
- Position sizing based on edge and volatility

**Financial Engineering:**
- Utility-based derivatives pricing
- Portfolio construction with constraints
- Multi-period optimization under uncertainty

## Usage Example

```python
from portfolio_gambling import SinglePeriodOptimizer, MultiPeriodOptimizer, GamblingStrategy
import numpy as np

# Generate returns
returns = np.random.multivariate_normal(mean, cov, 1000)

# Single-period optimization
sp_opt = SinglePeriodOptimizer(returns)
mv_weights = sp_opt.mean_variance()  # Tangency portfolio
util_weights = sp_opt.expected_utility('log')  # Log utility

# Multi-period optimization  
mp_opt = MultiPeriodOptimizer(returns)
kelly_weights = mp_opt.kelly_criterion()
results = mp_opt.simulate_strategy(kelly_weights, n_periods=252)

# Gambling strategy
game = GamblingStrategy(initial_wealth=50, target_wealth=100, 
                       win_prob=0.474, win_odds=1.0)
comparison = game.compare_strategies()
```

## Dependencies

- numpy: Numerical computations
- scipy: Optimization algorithms
- pandas: Data manipulation
- matplotlib: Plotting
- seaborn: Statistical visualizations

## Academic Foundations

**Key Papers Implemented:**
1. Markowitz (1952): "Portfolio Selection"
2. Roy (1952): "Safety First and the Holding of Assets"
3. Kelly (1956): "A New Interpretation of Information Rate"
4. Dubins & Savage (1965): "How to Gamble If You Must"
5. Merton (1969): "Lifetime Portfolio Selection"

## Target Audience

- Quantitative finance professionals
- Academic researchers in portfolio theory
- Students learning portfolio optimization
- Traders interested in position sizing
- Anyone exploring connections between gambling theory and portfolio management

## Advanced Features

- Regime-switching market simulations
- Dynamic Kelly with rolling estimation
- Transaction cost modeling
- Turnover analysis
- Certainty equivalent calculations
- Growth rate comparisons
- Leverage analysis

This package bridges theory and practice, providing both rigorous implementations and intuitive visualizations of classic portfolio optimization and gambling theory results.
