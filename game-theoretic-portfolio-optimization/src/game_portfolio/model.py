"""Portfolio-game model and equilibrium residuals.

The implementation uses d=n-1 free logits and fixes the final logit to zero.
This removes the additive softmax invariance and gives a nonsingular local
parameterization of the interior simplex.
"""

from dataclasses import dataclass
import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    y = x - np.max(x)
    e = np.exp(y)
    return e / e.sum()


def _weights_and_jacobian(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map n-1 logits to n weights and return the reduced Jacobian."""
    z = np.asarray(z, dtype=float)
    w = softmax(np.r_[z, 0.0])
    full_j = np.diag(w) - np.outer(w, w)
    return w, full_j[:, :-1]


@dataclass
class PortfolioGame:
    """Two-player portfolio game.

    Parameters
    ----------
    mu_types:
        Array with shape (2, n_types, n_assets). Entry [i, t] is player i's
        expected-return vector when that player's private type is t.
    covariance:
        Positive-definite covariance matrix.
    risk_aversion:
        Length-two vector of risk-aversion coefficients.
    concentration:
        Length-two vector for a smooth quadratic concentration penalty.
    crowding:
        Length-two vector multiplying w_i.T @ interaction @ w_j.
    interaction:
        Matrix describing which assets have crowding interaction.
    joint_types:
        Joint probability table for (type_1, type_2). Rows sum to one.
    """

    mu_types: np.ndarray
    covariance: np.ndarray
    risk_aversion: np.ndarray = None
    concentration: np.ndarray = None
    crowding: np.ndarray = None
    interaction: np.ndarray = None
    joint_types: np.ndarray = None

    def __post_init__(self):
        self.mu_types = np.asarray(self.mu_types, dtype=float)
        self.covariance = np.asarray(self.covariance, dtype=float)
        if self.mu_types.ndim != 3 or self.mu_types.shape[0] != 2:
            raise ValueError("mu_types must have shape (2, n_types, n_assets)")
        self.n_types = self.mu_types.shape[1]
        self.n_assets = self.mu_types.shape[2]
        if self.covariance.shape != (self.n_assets, self.n_assets):
            raise ValueError("covariance has incompatible shape")
        self.risk_aversion = np.ones(2) if self.risk_aversion is None else np.asarray(self.risk_aversion, float)
        self.concentration = np.zeros(2) if self.concentration is None else np.asarray(self.concentration, float)
        self.crowding = np.zeros(2) if self.crowding is None else np.asarray(self.crowding, float)
        self.interaction = np.eye(self.n_assets) if self.interaction is None else np.asarray(self.interaction, float)
        self.joint_types = np.ones((self.n_types, self.n_types)) / self.n_types**2 if self.joint_types is None else np.asarray(self.joint_types, float)
        if not np.isclose(self.joint_types.sum(), 1.0):
            raise ValueError("joint_types must sum to one")

    @property
    def dimension(self) -> int:
        return self.n_assets - 1

    def weights(self, z: np.ndarray) -> np.ndarray:
        return _weights_and_jacobian(z)[0]

    def utility(self, player: int, wi: np.ndarray, wj: np.ndarray, own_type: int) -> float:
        mu = self.mu_types[player, own_type]
        return float(mu @ wi - 0.5 * self.risk_aversion[player] * wi @ self.covariance @ wi
                     - 0.5 * self.concentration[player] * (wi @ wi)
                     - self.crowding[player] * wi @ self.interaction @ wj)

    def utility_gradient_weights(self, player: int, wi: np.ndarray, wj: np.ndarray, own_type: int) -> np.ndarray:
        return (self.mu_types[player, own_type]
                - self.risk_aversion[player] * self.covariance @ wi
                - self.concentration[player] * wi
                - self.crowding[player] * self.interaction @ wj)

    def utility_gradient_logits(self, player: int, zi: np.ndarray, zj: np.ndarray, own_type: int) -> np.ndarray:
        wi, jac = _weights_and_jacobian(zi)
        wj = self.weights(zj)
        return jac.T @ self.utility_gradient_weights(player, wi, wj, own_type)

    def complete_residual(self, x: np.ndarray, types: tuple[int, int] = (0, 0)) -> np.ndarray:
        """FOC residual for a complete-information equilibrium."""
        d = self.dimension
        z1, z2 = x[:d], x[d:]
        return np.r_[self.utility_gradient_logits(0, z1, z2, types[0]),
                     self.utility_gradient_logits(1, z2, z1, types[1])]

    def incomplete_residual(self, x: np.ndarray) -> np.ndarray:
        """Bayesian-Nash FOC residual for type-contingent strategies."""
        d, m = self.dimension, self.n_types
        strategies = x.reshape(2, m, d)
        result = np.zeros_like(strategies)
        row = self.joint_types.sum(axis=1)
        col = self.joint_types.sum(axis=0)
        for t1 in range(m):
            z1 = strategies[0, t1]
            for t2 in range(m):
                result[0, t1] += self.joint_types[t1, t2] / row[t1] * self.utility_gradient_logits(0, z1, strategies[1, t2], t1)
        for t2 in range(m):
            z2 = strategies[1, t2]
            for t1 in range(m):
                result[1, t2] += self.joint_types[t1, t2] / col[t2] * self.utility_gradient_logits(1, z2, strategies[0, t1], t2)
        return result.ravel()

    def best_response_gradient(self, player: int, zi: np.ndarray, zj: np.ndarray, own_type: int) -> np.ndarray:
        return self.utility_gradient_logits(player, zi, zj, own_type)
