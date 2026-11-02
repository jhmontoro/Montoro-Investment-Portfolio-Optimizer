"""Small, transparent implementations of numerical equilibrium solvers."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SolverResult:
    x: np.ndarray
    converged: bool
    iterations: int
    residual_norm: float
    history: list[float] = field(default_factory=list)


def finite_difference_jacobian(fun, x, eps=1e-6):
    x = np.asarray(x, float)
    f0 = np.asarray(fun(x), float)
    J = np.empty((f0.size, x.size))
    for j in range(x.size):
        h = eps * max(1.0, abs(x[j]))
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (fun(xp) - fun(xm)) / (2.0 * h)
    return J


def _line_search(fun, x, step, f=None):
    f = fun(x) if f is None else f
    phi = 0.5 * f @ f
    alpha = 1.0
    for _ in range(30):
        candidate = x + alpha * step
        if 0.5 * fun(candidate) @ fun(candidate) < phi:
            return candidate
        alpha *= 0.5
    return x


def newton(fun, x0, tol=1e-7, max_iter=100):
    x = np.asarray(x0, float).copy(); history = []
    for k in range(max_iter):
        f = np.asarray(fun(x), float); norm = np.linalg.norm(f); history.append(norm)
        if norm < tol: return SolverResult(x, True, k, norm, history)
        J = finite_difference_jacobian(fun, x)
        step = np.linalg.lstsq(J, -f, rcond=None)[0]
        x_new = _line_search(fun, x, step, f)
        if np.allclose(x_new, x): break
        x = x_new
    norm = np.linalg.norm(fun(x)); return SolverResult(x, norm < tol, len(history), norm, history)


def broyden(fun, x0, tol=1e-7, max_iter=200):
    x = np.asarray(x0, float).copy(); f = np.asarray(fun(x), float); B = finite_difference_jacobian(fun, x); history = []
    for k in range(max_iter):
        norm = np.linalg.norm(f); history.append(norm)
        if norm < tol: return SolverResult(x, True, k, norm, history)
        step = np.linalg.lstsq(B, -f, rcond=None)[0]
        x_new = _line_search(fun, x, step, f); f_new = np.asarray(fun(x_new), float)
        # Good Broyden can lose a useful secant model on a nonlinear
        # Bayesian system.  A fresh finite-difference Jacobian is a safe
        # globalization fallback when the proposed step makes no progress.
        if np.linalg.norm(f_new) > 0.999 * norm:
            B = finite_difference_jacobian(fun, x)
            step = np.linalg.lstsq(B, -f, rcond=None)[0]
            x_new = _line_search(fun, x, step, f)
            f_new = np.asarray(fun(x_new), float)
        s, y = x_new - x, f_new - f
        if s @ s > 1e-16: B = B + np.outer(y - B @ s, s) / (s @ s)
        x, f = x_new, f_new
    norm = np.linalg.norm(f); return SolverResult(x, norm < tol, len(history), norm, history)


def _objective(fun, x):
    f = np.asarray(fun(x), float); return 0.5 * f @ f


def _objective_gradient(fun, x):
    f = np.asarray(fun(x), float); return finite_difference_jacobian(fun, x).T @ f


def bfgs(fun, x0, tol=1e-7, max_iter=200):
    """BFGS on the nonlinear least-squares merit function 1/2||F(x)||^2."""
    x = np.asarray(x0, float).copy(); n = x.size
    # The residuals in portfolio games can be numerically small even when the
    # logits are not close to equilibrium.  A Gauss--Newton-scaled initial
    # inverse Hessian avoids the very small first steps produced by H=I.
    J0 = finite_difference_jacobian(fun, x)
    H = np.linalg.pinv(J0.T @ J0 + 1e-10 * np.eye(n))
    history = []
    for k in range(max_iter):
        f = np.asarray(fun(x), float); norm = np.linalg.norm(f); history.append(norm)
        if norm < tol: return SolverResult(x, True, k, norm, history)
        g = _objective_gradient(fun, x); p = -H @ g
        x_new = _line_search(lambda y: np.asarray(fun(y), float), x, p, f)
        s = x_new - x
        if np.linalg.norm(s) < 1e-14: break
        y = _objective_gradient(fun, x_new) - g; sy = s @ y
        if sy > 1e-12:
            rho = 1.0 / sy; I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = x_new
    norm = np.linalg.norm(fun(x)); return SolverResult(x, norm < tol, len(history), norm, history)


def trust_region(fun, x0, tol=1e-7, max_iter=200, radius0=1.0, radius_max=100.0):
    """Trust-region Gauss-Newton method with Cauchy fallback and rho updates."""
    x = np.asarray(x0, float).copy(); radius = radius0; history = []
    for k in range(max_iter):
        f = np.asarray(fun(x), float); phi = 0.5 * f @ f; norm = np.linalg.norm(f); history.append(norm)
        if norm < tol: return SolverResult(x, True, k, norm, history)
        J = finite_difference_jacobian(fun, x); g = J.T @ f; A = J.T @ J
        try: gn = np.linalg.lstsq(A, -g, rcond=None)[0]
        except np.linalg.LinAlgError: gn = -g
        if np.linalg.norm(gn) <= radius:
            step = gn
        else:
            gAg = g @ A @ g
            alpha = (g @ g) / gAg if gAg > 1e-16 else 1.0
            cauchy = -alpha * g
            if np.linalg.norm(cauchy) >= radius:
                step = -radius * g / max(np.linalg.norm(g), 1e-16)
            else:
                d = gn - cauchy; aa = d @ d; bb = 2 * cauchy @ d; cc = cauchy @ cauchy - radius**2
                tau = (-bb + np.sqrt(max(bb * bb - 4 * aa * cc, 0.0))) / (2 * aa) if aa > 1e-16 else 0.0
                step = cauchy + tau * d
        predicted = -(g @ step + 0.5 * step @ A @ step)
        actual = phi - _objective(fun, x + step)
        rho = actual / predicted if predicted > 1e-16 else -np.inf
        if rho > 0.1: x = x + step
        if rho < 0.25: radius *= 0.25
        elif rho > 0.75 and abs(np.linalg.norm(step) - radius) < 1e-8: radius = min(2 * radius, radius_max)
        if radius < 1e-12: break
    norm = np.linalg.norm(fun(x)); return SolverResult(x, norm < tol, len(history), norm, history)


def best_response_iteration(game, z0, types=(0, 0), tol=1e-8, max_iter=200, step0=1.0):
    """Alternating gradient-ascent best responses for complete information."""
    d = game.dimension; z = np.asarray(z0, float).reshape(2, d).copy(); history = []
    for k in range(max_iter):
        old = z.copy()
        for i, j in ((0, 1), (1, 0)):
            grad = game.best_response_gradient(i, z[i], z[j], types[i]); alpha = step0
            base = game.utility(i, game.weights(z[i]), game.weights(z[j]), types[i])
            while alpha > 1e-10:
                candidate = z[i] + alpha * grad
                value = game.utility(i, game.weights(candidate), game.weights(z[j]), types[i])
                if value >= base: z[i] = candidate; break
                alpha *= 0.5
        residual = np.linalg.norm(game.complete_residual(z.ravel(), types)); history.append(residual)
        if np.linalg.norm(z - old) < tol: break
    residual = np.linalg.norm(game.complete_residual(z.ravel(), types))
    return SolverResult(z.ravel(), residual < 1e-5, len(history), residual, history)
