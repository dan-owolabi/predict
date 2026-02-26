"""
Dixon-Coles goal model for football prediction.
Estimates team attack/defense ratings via MLE with time-decay weighting.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from math import exp, log


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


def dc_log_likelihood(params, hi, ai, hg, ag, w, n_teams):
    """Negative log-likelihood for Dixon-Coles model (fully vectorized)."""
    attack = params[:n_teams]
    defense = params[n_teams:2*n_teams]
    home_adv = params[2*n_teams]
    rho = params[2*n_teams + 1]

    lam = np.exp(attack[hi] + defense[ai] + home_adv)
    mu = np.exp(attack[ai] + defense[hi])

    lam = np.maximum(lam, 0.001)
    mu = np.maximum(mu, 0.001)

    # Poisson log-likelihoods
    ll_home = poisson.logpmf(hg, lam)
    ll_away = poisson.logpmf(ag, mu)

    # Vectorized tau correction for low-scoring matches
    hg_int = hg.astype(int)
    ag_int = ag.astype(int)
    tau_vals = np.ones(len(hg))
    m00 = (hg_int == 0) & (ag_int == 0)
    m01 = (hg_int == 0) & (ag_int == 1)
    m10 = (hg_int == 1) & (ag_int == 0)
    m11 = (hg_int == 1) & (ag_int == 1)
    tau_vals[m00] = 1 - lam[m00] * mu[m00] * rho
    tau_vals[m01] = 1 + lam[m01] * rho
    tau_vals[m10] = 1 + mu[m10] * rho
    tau_vals[m11] = 1 - rho
    tau_vals = np.maximum(tau_vals, 1e-10)

    ll = np.sum(w * (ll_home + ll_away + np.log(tau_vals)))

    # Light L2 regularization (just enough to prevent drift, not kill signal)
    ll -= 0.001 * np.sum(attack**2)
    ll -= 0.001 * np.sum(defense**2)

    # Sum-to-zero constraint on attack params (proper identifiability)
    ll -= 10.0 * attack.sum()**2

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

        # Prepare match data as numpy arrays
        hi = np.array([self.teams[h] for h in home_teams])
        ai = np.array([self.teams[a] for a in away_teams])
        hg = np.array(home_goals, dtype=float)
        ag = np.array(away_goals, dtype=float)
        w = np.array(weights, dtype=float)

        # Initial params: attack, defense, home_adv, rho
        x0 = np.zeros(2 * n + 2)
        x0[2*n] = 0.25     # home advantage
        x0[2*n + 1] = -0.05  # rho (small negative typical)

        # Bounds
        bounds = [(-2.0, 2.0)] * n          # attack
        bounds += [(-2.0, 2.0)] * n         # defense
        bounds += [(0.0, 1.0)]              # home_adv
        bounds += [(-0.3, 0.3)]             # rho

        result = minimize(
            dc_log_likelihood,
            x0,
            args=(hi, ai, hg, ag, w, n),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )

        # Store results
        for team, idx in self.teams.items():
            self.attack[team] = result.x[idx]
            self.defense[team] = result.x[n + idx]
        self.home_adv = result.x[2*n]
        self.rho = result.x[2*n + 1]

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
        total = probs.sum()
        if total > 0:
            probs /= total
        return probs

    def predict_ou15(self, home, away):
        """P(Over 1.5 goals)."""
        probs = self.predict_score_probs(home, away)
        p_under = sum(probs[i, j] for i in range(9) for j in range(9) if i+j <= 1)
        return 1.0 - p_under

    def predict_ou25(self, home, away):
        """P(Over 2.5 goals)."""
        probs = self.predict_score_probs(home, away)
        p_under = sum(probs[i, j] for i in range(9) for j in range(9) if i+j <= 2)
        return 1.0 - p_under

    def predict_btts(self, home, away):
        """P(Both Teams To Score)."""
        probs = self.predict_score_probs(home, away)
        p_no_btts = sum(probs[0, :]) + sum(probs[:, 0]) - probs[0, 0]
        return 1 - p_no_btts

    def predict_match_result(self, home, away):
        """P(Home), P(Draw), P(Away)."""
        probs = self.predict_score_probs(home, away)
        p_home = sum(probs[i, j] for i in range(9) for j in range(9) if i > j)
        p_draw = sum(probs[i, i] for i in range(9))
        p_away = sum(probs[i, j] for i in range(9) for j in range(9) if i < j)
        return p_home, p_draw, p_away

    def predict_top_scores(self, home, away, n=3):
        """Return top N most likely scorelines with probabilities."""
        probs = self.predict_score_probs(home, away)
        scores = []
        for i in range(9):
            for j in range(9):
                if probs[i, j] > 0.001:
                    scores.append((f"{i}-{j}", probs[i, j]))
        scores.sort(key=lambda x: -x[1])
        return scores[:n]
