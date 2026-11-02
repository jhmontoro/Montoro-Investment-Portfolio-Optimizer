"""Run the complete- and incomplete-information solver comparison.

Usage from the repository root:

    PYTHONPATH=src python examples/run_experiments.py
"""

import numpy as np

from game_portfolio import PortfolioGame, newton, broyden, bfgs, trust_region


def make_demo_game():
    expected_returns = np.array([
        [[0.08, 0.10, 0.04], [0.03, 0.01, 0.06]],
        [[0.07, 0.09, 0.05], [0.02, 0.02, 0.07]],
    ])
    covariance = np.array([
        [0.04, 0.01, 0.00],
        [0.01, 0.05, 0.01],
        [0.00, 0.01, 0.01],
    ])
    return PortfolioGame(
        expected_returns,
        covariance,
        risk_aversion=np.array([3.0, 3.0]),
        concentration=np.array([0.05, 0.05]),
        crowding=np.array([0.02, 0.02]),
        joint_types=np.array([[0.35, 0.15], [0.15, 0.35]]),
    )


def run():
    game = make_demo_game()
    solvers = (newton, broyden, bfgs, trust_region)
    complete_x0 = np.zeros(2 * game.dimension)
    incomplete_x0 = np.zeros(2 * game.n_types * game.dimension)

    print("Complete-information equilibrium")
    complete_fun = lambda x: game.complete_residual(x, types=(0, 0))
    for solver in solvers:
        result = solver(complete_fun, complete_x0, max_iter=500)
        print(f"{solver.__name__:>14}: converged={result.converged!s:<5} "
              f"iterations={result.iterations:>4} residual={result.residual_norm:.3e}")

    print("\nIncomplete-information Bayesian Nash equilibrium")
    for solver in solvers:
        result = solver(game.incomplete_residual, incomplete_x0, max_iter=500)
        print(f"{solver.__name__:>14}: converged={result.converged!s:<5} "
              f"iterations={result.iterations:>4} residual={result.residual_norm:.3e}")


if __name__ == "__main__":
    run()
