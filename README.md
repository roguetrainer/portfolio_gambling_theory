# Portfolio Management and Gambling Theory

A comprehensive Python package exploring portfolio optimization criteria across single-period and infinite-horizon settings, with connections to optimal gambling strategies from Dubins & Savage's classic "How to Gamble If You Must" (1965).

![Portfolio-gambling](./img/Portfolio-gambling.png)

## Overview

This package implements and compares different approaches to portfolio selection and gambling strategies:

**Single-Period Criteria:**
- **Mean-Variance** (Markowitz 1952): Classic risk-return tradeoff
- **Expected Utility**: CRRA, CARA, and logarithmic utility maximization
- **VaR/CVaR**: Tail risk minimization
- **Safety-First** (Roy 1952): Disaster avoidance

**Multi-Period/Infinite Horizon:**
- **Kelly Criterion** (1956): Growth-optimal betting
- **Merton Portfolio** (1969): Continuous-time optimization
- **Dynamic Programming**: Finite-horizon optimal control
- **Bold vs. Timid Play** (Dubins-Savage 1965): Subfair game strategies
---
![How to Gamble](./img/HTGIYM.jpg)
---
## Quick Start

```bash
# Install package
pip install -r requirements.txt
pip install -e .

# Run example
python examples/quick_start.py

# Explore notebooks
cd notebooks
jupyter notebook
```

## Full documentation

See `/docs/` directory for:
- [`theoretical_foundations.md`](./docs/theoretical_foundations.md): Mathematical background
- [`implementation_guide.md`](./docs/implementation_guide.md): Detailed usage guide
- Example notebooks in `/notebooks/`
