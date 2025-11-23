# Theoretical Foundations: Portfolio Management and Gambling Theory

## Overview

This document provides the mathematical foundations for portfolio optimization criteria and their connections to optimal gambling strategies.

## Single-Period Criteria

### 1. Mean-Variance (Markowitz, 1952)

**Objective**: Minimize portfolio variance for a given expected return, or maximize:
```
max_w  μ'w - (λ/2)w'Σw
s.t.   1'w = 1
       w ≥ 0  (if no short sales)
```

Where:
- w = portfolio weights (n x 1)
- μ = expected returns (n x 1)
- Σ = covariance matrix (n x n)
- λ = risk aversion parameter

**Closed-form solution** (unconstrained):
```
w* = (1/λ)Σ^(-1)μ + ν Σ^(-1)1
```
where ν ensures weights sum to 1.

**Key properties**:
- Optimal for quadratic utility: U(W) = W - (λ/2)W²
- Assumes returns are normally distributed (or investor has quadratic utility)
- Efficient frontier is hyperbola in (σ, μ) space
- Symmetric treatment of upside/downside risk

### 2. Expected Utility

**Objective**: Maximize E[U(W_final)]

Common utility functions:

**CRRA (Constant Relative Risk Aversion)**:
```
U(W) = W^(1-γ)/(1-γ)  for γ ≠ 1
U(W) = log(W)          for γ = 1
```

**CARA (Constant Absolute Risk Aversion)**:
```
U(W) = -exp(-αW)
```

**Properties**:
- γ (or α) measures risk aversion
- Higher γ → more conservative
- γ = 1 (log utility) leads to Kelly criterion in multi-period setting
- No closed-form solution in general; requires numerical optimization

**Arrow-Pratt measures**:
- Absolute risk aversion: A(W) = -U''(W)/U'(W)
- Relative risk aversion: R(W) = -WU''(W)/U'(W)

### 3. Value-at-Risk (VaR) and CVaR

**VaR_α** = smallest loss x such that P(Loss > x) ≤ α

**CVaR_α** (Conditional VaR or Expected Shortfall):
```
CVaR_α = E[Loss | Loss ≥ VaR_α]
```

**Optimization**:
```
min_w  CVaR_α(r'w)
s.t.   μ'w ≥ r_target
       1'w = 1
       w ≥ 0
```

**Properties**:
- CVaR is a coherent risk measure (VaR is not)
- CVaR optimization can be solved as LP
- Focus on tail risk rather than variance
- Popular in risk management and regulation

### 4. Safety-First (Roy, 1952)

**Objective**: Maximize "safety-first ratio":
```
max_w  (μ'w - d) / √(w'Σw)
```
where d is the "disaster level" (minimum acceptable return).

**Interpretation**:
- Under normality, minimizes P(return < d)
- Equivalent to maximizing distance to disaster in standard deviations
- Similar to Sharpe ratio when d = risk-free rate

## Multi-Period / Infinite Horizon Criteria

### 1. Kelly Criterion (1956)

**Objective**: Maximize long-run growth rate:
```
max_w  E[log(1 + r'w)]
```

**For continuous returns** (approximately):
```
w* = Σ^(-1)μ
```

**For discrete binary bet** with probability p, odds b:1:
```
f* = (bp - q)/b  where q = 1-p
```

**Properties**:
- Maximizes E[log(wealth)] ⟹ maximizes median wealth
- Asymptotically optimal: highest growth rate almost surely
- Can be very volatile in short/medium term
- Equivalent to log utility (γ = 1)
- "Time diversification" argument

**Fractional Kelly**:
```
f = (fraction) × f*
```
Reduces volatility at cost of growth rate.

### 2. Merton's Continuous-Time Portfolio (1969, 1971)

**Problem**: Maximize expected utility of terminal wealth and/or consumption over time.

**HJB Equation**:
```
ρV(W,t) = max_w,c {U(c) + V_t + V_W(μ'w W - c) + (1/2)V_WW W²w'Σw}
```

**For power utility** U(W) = W^(1-γ)/(1-γ) and constant opportunities:
```
w* = (1/γ)Σ^(-1)(μ - r1)
```

**Properties**:
- Myopic for log and power utility: don't need to hedge against changing opportunities
- Non-myopic for other utilities or time-varying opportunities
- Hedging demands arise when investment opportunities vary with state variables

### 3. Dynamic Programming / Bellman Optimality

**Value function**:
```
V(W,t) = max_{strategy} E[U(W_T) | W_t = W]
```

**Bellman equation**:
```
V(W,t) = max_w E[V(W_{t+1}, t+1) | W_t = W]
```

**Properties**:
- Handles time-varying investment opportunities
- Can incorporate labor income, taxes, constraints
- Generally requires numerical solution (grid methods, etc.)

## Gambling Strategies (Dubins & Savage, 1965)

### Bold Play vs. Timid Play

**Setup**: Start with wealth w, target W, subfair game (p < 0.5).

**Bold Play**: Bet maximum each round:
```
bet = min(w, W - w)
```

**Theorem**: For subfair games, bold play maximizes P(reach W before 0).

**Success probability** (bold play):
```
P_bold(w) = (w/W)^α  where α = log(q)/log(q/p)
```

**Timid Play**: Bet minimum each round. Suboptimal for subfair games.

**Key insight**: In subfair games, you want to minimize number of bets (exposure to negative expectation).

### Comparison to Kelly

**Kelly**: Maximizes growth rate, assumes favorable game (p > 0.5)
```
f* = (p - q)/1 = 2p - 1
```

**Bold Play**: Maximizes P(success), assumes subfair game (p < 0.5)
```
f* = 1  (bet everything)
```

**Key difference**:
- Kelly: Infinite horizon, maximize E[log W]
- Bold Play: Finite horizon with specific target, maximize P(reach target)

### Proportional Betting

**Strategy**: Bet fraction f of wealth each round.

**Growth rate** (for binary bet):
```
g(f) = p log(1 + fb) + (1-p) log(1 - f)
```

**Optimal** f = f_Kelly maximizes g(f).

**Properties**:
- Smooth wealth path
- Never go broke if f < 1
- Suboptimal for reaching specific target in subfair game

## Connections and Trade-offs

### Mean-Variance vs. Expected Utility
- MV is approximation to EU when returns normal or utility quadratic
- EU handles skewness, kurtosis, non-normality
- MV is computationally simpler

### Kelly vs. Mean-Variance
- Kelly ≈ MV with γ = 1 (log utility)
- Kelly focuses on growth rate, ignores interim volatility
- MV can accommodate different risk aversions

### Bold Play vs. Kelly
- Bold: finite horizon, specific target, subfair game
- Kelly: infinite horizon, growth optimization, favorable game
- Bold is "desperate" strategy when odds against you

### Single-Period vs. Multi-Period
- Single period: ignores sequential nature of investing
- Multi-period: captures rebalancing, changing opportunities
- With i.i.d. returns and myopic utility: reduce to single-period problem

## Practical Considerations

### Transaction Costs
- Add penalty term for turnover
- Creates "no-trade region" around optimal weights
- Reduces rebalancing frequency

### Parameter Uncertainty
- Sample estimates μ̂, Σ̂ have estimation error
- Leads to "estimation risk"
- Solutions: Bayesian methods, robust optimization, shrinkage

### Constraints
- No short sales: w ≥ 0
- Position limits: w_i ≤ u_i
- Leverage limits: ||w||_1 ≤ L
- Factor exposures
- Turn convex optimization into more complex problem

### Implementation
- Mean-variance: Quadratic programming (QP)
- Expected utility: Nonlinear optimization
- CVaR: Linear programming (LP)
- Kelly/log utility: Often reduces to QP or closed form

## Historical Context

1. **Markowitz (1952)**: Modern portfolio theory foundation
2. **Roy (1952)**: Safety-first, alternative to MV
3. **Kelly (1956)**: Information theory applied to gambling
4. **Dubins & Savage (1965)**: Rigorous gambling theory
5. **Merton (1969, 1971)**: Continuous-time portfolio theory
6. **Thorp (1969, 2006)**: Practical Kelly applications

## Key References

1. Markowitz, H. (1952). "Portfolio Selection", Journal of Finance
2. Roy, A. D. (1952). "Safety First and the Holding of Assets", Econometrica
3. Kelly, J. (1956). "A New Interpretation of Information Rate"
4. Dubins, L. E., & Savage, L. J. (1965). "How to Gamble If You Must"
5. Merton, R. C. (1969). "Lifetime Portfolio Selection under Uncertainty"
6. Samuelson, P. A. (1979). "Why We Should Not Make Mean Log of Wealth Big"
7. Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports Betting and the Stock Market"
8. Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of CVaR"

## Summary Table

| Criterion | Objective | Risk Measure | Horizon | Best For |
|-----------|-----------|--------------|---------|----------|
| Mean-Variance | max μ - λσ² | Variance | Single | Normal returns, quadratic utility |
| Expected Utility | max E[U(W)] | Utility-based | Any | General preferences |
| VaR/CVaR | min tail risk | Quantile | Single | Risk management |
| Safety-First | max (μ-d)/σ | Disaster prob | Single | Avoid specific loss |
| Kelly | max E[log W] | Growth rate | Infinite | Long-term growth |
| Merton | max E[∫U(c)dt] | Utility + time | Continuous | Consumption-investment |
| Bold Play | max P(reach W) | Ruin prob | Finite | Desperate situations |
