"""Game-theoretic portfolio optimization."""

from .model import PortfolioGame, softmax
from .solvers import (
    SolverResult,
    newton,
    broyden,
    bfgs,
    trust_region,
    best_response_iteration,
)

__all__ = [
    "PortfolioGame",
    "softmax",
    "SolverResult",
    "newton",
    "broyden",
    "bfgs",
    "trust_region",
    "best_response_iteration",
]
