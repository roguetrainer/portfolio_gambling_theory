"""
Portfolio Gambling Theory Package

A comprehensive package for portfolio optimization and gambling theory.
"""

from .single_period import SinglePeriodOptimizer
from .multi_period import MultiPeriodOptimizer
from .gambling_strategies import GamblingStrategy

__version__ = '0.1.0'
__all__ = ['SinglePeriodOptimizer', 'MultiPeriodOptimizer', 'GamblingStrategy']
