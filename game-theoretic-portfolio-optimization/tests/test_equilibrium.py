import numpy as np

from game_portfolio import PortfolioGame, softmax, newton, broyden, bfgs, trust_region


def make_game():
    mu = np.array([
        [[0.08, 0.10, 0.04], [0.03, 0.01, 0.06]],
        [[0.07, 0.09, 0.05], [0.02, 0.02, 0.07]],
    ])
    cov = np.array([[0.04, 0.01, 0.00], [0.01, 0.05, 0.01], [0.00, 0.01, 0.01]])
    return PortfolioGame(
        mu, cov, risk_aversion=np.array([3.0, 3.0]),
        concentration=np.array([0.05, 0.05]), crowding=np.array([0.02, 0.02]),
        joint_types=np.array([[0.35, 0.15], [0.15, 0.35]]),
    )


def test_softmax_is_simplex():
    w = softmax(np.array([1000.0, 1001.0, 999.0]))
    assert np.all(w > 0)
    assert np.isclose(w.sum(), 1.0)


def test_complete_residual_solvers():
    game = make_game()
    fun = lambda x: game.complete_residual(x, (0, 0))
    x0 = np.zeros(2 * game.dimension)
    for solver in (newton, broyden, bfgs, trust_region):
        result = solver(fun, x0, max_iter=300)
        assert result.residual_norm < 1e-5, (solver.__name__, result.residual_norm)
        weights = result.x.reshape(2, game.dimension)
        for z in weights:
            assert np.isclose(game.weights(z).sum(), 1.0)


def test_incomplete_information_bayesian_residual():
    game = make_game()
    x0 = np.zeros(2 * game.n_types * game.dimension)
    result = trust_region(game.incomplete_residual, x0, max_iter=300)
    assert result.residual_norm < 1e-5
    strategies = result.x.reshape(2, game.n_types, game.dimension)
    for player in range(2):
        for typ in range(game.n_types):
            assert np.isclose(game.weights(strategies[player, typ]).sum(), 1.0)
