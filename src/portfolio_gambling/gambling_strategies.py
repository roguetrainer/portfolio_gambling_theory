"""
Gambling Strategies from Dubins & Savage

Implements optimal gambling strategies for subfair games:
- Bold Play (bet maximum allowed)
- Timid Play (bet minimum allowed)
- Optimal strategy for reaching target before ruin
"""

import numpy as np
from scipy.stats import binom


class GamblingStrategy:
    """Gambling strategies from 'How to Gamble If You Must'."""
    
    def __init__(self, initial_wealth, target_wealth, win_prob, win_odds=1.0):
        """
        Initialize gambling problem.
        
        Parameters:
        -----------
        initial_wealth : float
            Starting wealth
        target_wealth : float
            Target wealth to reach
        win_prob : float
            Probability of winning a bet
        win_odds : float
            Odds: win amount per unit bet (default 1:1)
        """
        self.w0 = initial_wealth
        self.W = target_wealth
        self.p = win_prob
        self.q = 1 - win_prob
        self.b = win_odds  # Win b per unit bet
        
        # Check if game is fair, subfair, or superfair
        self.edge = self.p * self.b - self.q
        if abs(self.edge) < 1e-10:
            self.game_type = 'fair'
        elif self.edge < 0:
            self.game_type = 'subfair'
        else:
            self.game_type = 'superfair'
    
    def bold_play_probability(self):
        """
        Calculate probability of reaching target with bold play.
        
        Bold play: always bet min(current_wealth, target - current_wealth).
        
        For subfair games, bold play is optimal.
        
        Returns:
        --------
        prob : float
            Probability of reaching target before ruin
        """
        if self.game_type == 'subfair':
            # For subfair games with even odds (b=1)
            # P(reach target | start at w) = (w/W)^α where α = log(q)/log(p/q)
            if abs(self.b - 1.0) < 1e-6:
                if abs(self.p - 0.5) < 1e-10:
                    # Fair game
                    return self.w0 / self.W
                else:
                    # Subfair game
                    alpha = np.log(self.q / self.p) / np.log(self.q / self.p + 1)
                    return (self.w0 / self.W) ** alpha
            else:
                # General case: need to solve recursively
                return self._bold_play_recursive()
        else:
            # For fair or superfair games, calculate explicitly
            return self._bold_play_recursive()
    
    def _bold_play_recursive(self):
        """Calculate bold play probability via dynamic programming."""
        # Discretize wealth levels
        wealth_levels = self._get_wealth_levels_bold()
        n_levels = len(wealth_levels)
        
        # P[i] = probability of reaching target from wealth_levels[i]
        P = np.zeros(n_levels)
        
        # Boundary conditions
        P[-1] = 1.0  # Already at target
        P[0] = 0.0   # Ruined
        
        # Map wealth to index
        wealth_to_idx = {w: i for i, w in enumerate(wealth_levels)}
        
        # Work backwards from high wealth to low
        for i in range(n_levels - 2, 0, -1):
            w = wealth_levels[i]
            
            # Bold play: bet min(w, W-w)
            bet = min(w, self.W - w)
            
            # Win: reach w + bet*b
            # Lose: reach w - bet
            w_win = w + bet * self.b
            w_lose = w - bet
            
            # Find closest wealth levels
            idx_win = self._find_closest_index(wealth_levels, w_win, wealth_to_idx)
            idx_lose = self._find_closest_index(wealth_levels, w_lose, wealth_to_idx)
            
            P[i] = self.p * P[idx_win] + self.q * P[idx_lose]
        
        # Return probability from initial wealth
        idx_initial = self._find_closest_index(wealth_levels, self.w0, wealth_to_idx)
        return P[idx_initial]
    
    def timid_play_probability(self, bet_size=None):
        """
        Calculate probability of reaching target with timid play.
        
        Timid play: always bet small amount (minimum bet).
        
        Parameters:
        -----------
        bet_size : float, optional
            Fixed bet size. If None, uses 1% of initial wealth.
            
        Returns:
        --------
        prob : float
            Probability of reaching target before ruin
        """
        if bet_size is None:
            bet_size = self.w0 * 0.01
        
        # For small bets, approximate as continuous random walk
        # This is essentially gambler's ruin problem
        if abs(self.p - 0.5) < 1e-10:
            # Fair game
            return self.w0 / self.W
        else:
            # Subfair/superfair game
            q_over_p = self.q / self.p
            if abs(q_over_p - 1.0) > 1e-10:
                numerator = 1 - q_over_p ** (self.w0 / bet_size)
                denominator = 1 - q_over_p ** (self.W / bet_size)
                return numerator / denominator if abs(denominator) > 1e-10 else 0
            else:
                return self.w0 / self.W
    
    def simulate_bold_play(self, n_simulations=10000):
        """
        Monte Carlo simulation of bold play strategy.
        
        Returns:
        --------
        results : dict
            Success rate, average number of bets, wealth distribution
        """
        successes = 0
        num_bets = []
        final_wealth = []
        
        for _ in range(n_simulations):
            wealth = self.w0
            bets_made = 0
            
            while 0 < wealth < self.W and bets_made < 1000:
                # Bold play bet
                bet = min(wealth, self.W - wealth)
                
                # Outcome
                if np.random.random() < self.p:
                    wealth += bet * self.b
                else:
                    wealth -= bet
                
                bets_made += 1
            
            if wealth >= self.W:
                successes += 1
            
            num_bets.append(bets_made)
            final_wealth.append(wealth)
        
        return {
            'success_rate': successes / n_simulations,
            'mean_bets': np.mean(num_bets),
            'median_bets': np.median(num_bets),
            'final_wealth_dist': np.array(final_wealth)
        }
    
    def simulate_timid_play(self, bet_size=None, n_simulations=10000):
        """
        Monte Carlo simulation of timid play strategy.
        
        Parameters:
        -----------
        bet_size : float, optional
            Fixed bet size
        n_simulations : int
            Number of simulations
            
        Returns:
        --------
        results : dict
            Success rate, average number of bets, wealth distribution
        """
        if bet_size is None:
            bet_size = self.w0 * 0.01
        
        successes = 0
        num_bets = []
        final_wealth = []
        
        for _ in range(n_simulations):
            wealth = self.w0
            bets_made = 0
            
            while 0 < wealth < self.W and bets_made < 10000:
                # Timid play: fixed small bet
                actual_bet = min(bet_size, wealth, self.W - wealth)
                
                # Outcome
                if np.random.random() < self.p:
                    wealth += actual_bet * self.b
                else:
                    wealth -= actual_bet
                
                bets_made += 1
            
            if wealth >= self.W:
                successes += 1
            
            num_bets.append(bets_made)
            final_wealth.append(wealth)
        
        return {
            'success_rate': successes / n_simulations,
            'mean_bets': np.mean(num_bets),
            'median_bets': np.median(num_bets),
            'final_wealth_dist': np.array(final_wealth)
        }
    
    def compare_strategies(self, n_simulations=10000):
        """
        Compare bold vs timid play.
        
        Returns:
        --------
        comparison : dict
            Results for both strategies
        """
        bold_results = self.simulate_bold_play(n_simulations)
        timid_results = self.simulate_timid_play(n_simulations=n_simulations)
        
        # Theoretical probabilities
        bold_prob_theory = self.bold_play_probability()
        timid_prob_theory = self.timid_play_probability()
        
        return {
            'bold_play': {
                **bold_results,
                'theoretical_prob': bold_prob_theory
            },
            'timid_play': {
                **timid_results,
                'theoretical_prob': timid_prob_theory
            },
            'game_type': self.game_type,
            'edge': self.edge
        }
    
    def _get_wealth_levels_bold(self):
        """Generate wealth levels reachable by bold play."""
        levels = {0, self.W}
        current = {self.w0}
        
        for _ in range(100):  # Max iterations
            new_levels = set()
            for w in current:
                if 0 < w < self.W:
                    bet = min(w, self.W - w)
                    w_win = min(w + bet * self.b, self.W)
                    w_lose = max(w - bet, 0)
                    new_levels.add(w_win)
                    new_levels.add(w_lose)
            
            if new_levels.issubset(levels):
                break
            levels.update(new_levels)
            current = new_levels
        
        return sorted(list(levels))
    
    def _find_closest_index(self, levels, value, value_map):
        """Find index of closest wealth level."""
        if value in value_map:
            return value_map[value]
        
        # Find closest
        idx = np.argmin(np.abs(np.array(levels) - value))
        return idx
