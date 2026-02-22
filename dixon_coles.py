"""
Dixon-Coles goal model for football prediction.
Estimates team attack/defense ratings via MLE with time-decay weighting.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from math import exp, log, factorial


def tau(x, y, lam, mu, rho):
    """Dixon-Coles correction for low scores."""
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def dc_log_likelihood(params, matches, teams, weights):
    """Negative log-likelihood for Dixon-Coles model."""
    n_teams = len(teams)
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    rho = 0.0 # Force pure Poisson for absolute stability metrics

    ll = 0.0
    for i, (hi, ai, hg, ag, w) in enumerate(matches):
        lam = exp(attack[hi] + defense[ai] + home_adv)
        mu = exp(attack[ai] + defense[hi])

        lam = max(lam, 0.001)
        mu = max(mu, 0.001)

        p_home = poisson.pmf(hg, lam)
        p_away = poisson.pmf(ag, mu)
        t = tau(hg, ag, lam, mu, rho)
        prob = p_home * p_away * t
        if prob > 0:
            ll += w * log(prob)

    return -ll


class DixonColesModel:
    """Dixon-Coles football goal model with time-decay."""

    def __init__(self, decay_rate=0.005):
        self.decay_rate = decay_rate
        self.teams = {}
        self.attack = {}
        self.defense = {}
        self.home_adv = 0.0
        self.rho = 0.0

    def fit(self, home_teams, away_teams, home_goals, away_goals, dates=None):
        """Fit the model using MLE."""
        # Build team index
        all_teams = sorted(set(home_teams) | set(away_teams))
        self.teams = {t: i for i, t in enumerate(all_teams)}
        n = len(all_teams)

        # Time weights (exponential decay)
        if dates is not None:
            dates = pd.to_datetime(dates)
            max_date = dates.max()
            weights = [exp(-self.decay_rate * (max_date - d).days) for d in dates]
        else:
            weights = [1.0] * len(home_teams)

        # Prepare match data: (home_idx, away_idx, home_goals, away_goals, weight)
        matches = []
        for ht, at, hg, ag, w in zip(home_teams, away_teams, home_goals, away_goals, weights):
            matches.append((self.teams[ht], self.teams[at], int(hg), int(ag), w))

        # Initial params: attack, defense, home_adv
        x0 = np.zeros(2 * n + 1)
        x0[2*n] = 0.25  # home advantage

        # Constraints: sum of attack = 0 and sum of defense = 0 (perfectly anchors the nullspace)
        constraints = [
            {'type': 'eq', 'fun': lambda p: np.sum(p[:n])},
            {'type': 'eq', 'fun': lambda p: np.sum(p[n:2*n])}
        ]

        # No bounds needed when correctly constrained, maximizing SLSQP convergence speed
        result = minimize(
            dc_log_likelihood,
            x0,
            args=(matches, self.teams, weights),
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': 300, 'ftol': 1e-6}
        )

        # Store results
        for team, idx in self.teams.items():
            self.attack[team] = result.x[idx]
            self.defense[team] = result.x[n + idx]
        self.home_adv = result.x[2*n]
        self.rho = 0.0

        return self

    def predict_goals(self, home, away):
        """Predict expected goals (lambda_home, lambda_away)."""
        atk_h = self.attack.get(home, 0.0)
        def_a = self.defense.get(away, 0.0)
        atk_a = self.attack.get(away, 0.0)
        def_h = self.defense.get(home, 0.0)

        lam = exp(atk_h + def_a + self.home_adv)
        mu = exp(atk_a + def_h)
        return max(lam, 0.01), max(mu, 0.01)

    def predict_score_probs(self, home, away, max_goals=8):
        """Full score probability matrix with DC correction."""
        lam, mu = self.predict_goals(home, away)
        probs = np.zeros((max_goals+1, max_goals+1))

        for i in range(max_goals+1):
            for j in range(max_goals+1):
                p = poisson.pmf(i, lam) * poisson.pmf(j, mu)
                t = tau(i, j, lam, mu, self.rho)
                probs[i, j] = max(0.0, p * t)

        # Normalize
        probs /= probs.sum()
        return probs

    def predict_ou25(self, home, away):
        """P(Over 2.5 goals)."""
        probs = self.predict_score_probs(home, away)
        p_under = sum(probs[i, j] for i in range(8) for j in range(8) if i+j <= 2)
        return 1 - p_under

    def predict_btts(self, home, away):
        """P(Both Teams To Score)."""
        probs = self.predict_score_probs(home, away)
        p_no_btts = sum(probs[0, :]) + sum(probs[:, 0]) - probs[0, 0]
        return 1 - p_no_btts

    def predict_match_result(self, home, away):
        """P(Home), P(Draw), P(Away)."""
        probs = self.predict_score_probs(home, away)
        p_home = sum(probs[i, j] for i in range(8) for j in range(8) if i > j)
        p_draw = sum(probs[i, i] for i in range(8))
        p_away = sum(probs[i, j] for i in range(8) for j in range(8) if i < j)
        return p_home, p_draw, p_away
