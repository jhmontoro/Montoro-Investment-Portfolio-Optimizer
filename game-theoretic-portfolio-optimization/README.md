# Game-Theoretic Portfolio Optimization

A reproducible Math 693A project implementing numerical methods for complete- and incomplete-information portfolio games.

The project follows the progression:

1. single-agent portfolio utility;
2. complete-information Nash equilibrium;
3. Bayesian Nash equilibrium with private market types.

The implementation includes Newton, Broyden, BFGS, trust-region Gauss–Newton, and alternating best-response iteration. The model uses a softmax reparameterization so the long-only, fully-invested portfolio constraint is handled while the numerical algorithms operate on unconstrained logits.

## Installation

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## Minimal usage

```python
import numpy as np
from game_portfolio import PortfolioGame, newton, broyden, bfgs, trust_region

mu = np.array([
    [[0.08, 0.10, 0.04], [0.03, 0.01, 0.06]],
    [[0.07, 0.09, 0.05], [0.02, 0.02, 0.07]],
])
cov = np.array([[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.01]])
game = PortfolioGame(mu, cov, risk_aversion=np.array([3., 3.]), crowding=np.array([.02, .02]))
x0 = np.zeros(2 * game.dimension)
result = trust_region(lambda x: game.complete_residual(x, (0, 0)), x0)
print(result.converged, result.residual_norm)
```

For incomplete information, initialize a vector with shape `(2, n_types, n_assets-1)` and call `game.incomplete_residual` with any of the four solvers.

## Project experiments

The intended empirical study uses a small universe of liquid ETFs, rolling expected-return and covariance estimates, and out-of-sample evaluation. Compare equilibrium portfolios with equal weighting, risk parity, and a single-agent mean-variance baseline. Report residuals, evaluations, initialization sensitivity, covariance sensitivity, turnover, Sharpe ratio, volatility, and drawdown.

## Repository map

- `docs/derivation.md`: formal model, FOCs, complete-information equilibrium, and Bayesian Nash equilibrium.
- `src/game_portfolio/model.py`: portfolio game and residual functions.
- `src/game_portfolio/solvers.py`: numerical methods.
- `tests/`: automated mathematical and solver tests.
- `examples/run_experiments.py`: complete and incomplete information comparison.

Run the demonstration with:

```bash
PYTHONPATH=src python examples/run_experiments.py
```

The model is a research prototype, not investment advice or a production trading system.
