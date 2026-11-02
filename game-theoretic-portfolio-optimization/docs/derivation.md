# Mathematical derivation

## 1. Portfolio parameterization

For (n) assets, let (z_i\in\mathbb{R}^{n-1}) and define

\[
w_i(z_i)_k=\frac{\exp(\tilde z_{ik})}{\sum_{\ell=1}^n\exp(\tilde z_{i\ell})},
\qquad \tilde z_i=(z_i,0).
\]

Therefore (w_i\geq0) and (\mathbf 1^Tw_i=1). Fixing the final logit to zero removes the additive invariance of softmax.

## 2. Utility

For player (i\), with private type (\theta_i), define

\[
u_i(w_i,w_j;\theta_i)=\mu_i(\theta_i)^Tw_i-
\frac{\lambda_i}{2}w_i^T\Sigma w_i-
\frac{\gamma_i}{2}w_i^Tw_i-
\eta_iw_i^TQw_j.
\]

The weight-space gradient is

\[
\nabla_{w_i}u_i=\mu_i(\theta_i)-\lambda_i\Sigma w_i-\gamma_iw_i-\eta_iQw_j.
\]

If (J_i=\partial w_i/\partial z_i), then

\[
\nabla_{z_i}u_i=J_i^T\nabla_{w_i}u_i,
\qquad J_i=\left[\operatorname{diag}(w_i)-w_iw_i^T\right]_{[:,1:n-1]}.
\]

## 3. Complete-information Nash equilibrium

For known types ((\theta_1,\theta_2)), an interior equilibrium satisfies

\[
F(z_1,z_2)=
\begin{bmatrix}
\nabla_{z_1}u_1(w_1(z_1),w_2(z_2);\theta_1)\\
\nabla_{z_2}u_2(w_2(z_2),w_1(z_1);\theta_2)
\end{bmatrix}=0.
\]

The repository solves (F(x)=0) with Newton and Broyden, and minimizes (\phi(x)=\frac12\|F(x)\|^2) with BFGS and trust-region Gauss--Newton.

## 4. Bayesian Nash equilibrium

Let (p_{ab}=P(\theta_1=a,\theta_2=b)). Each player uses a type-contingent strategy (s_i(a)=z_i^a). Player 1's conditional first-order condition is

\[
\sum_bP(\theta_2=b\mid\theta_1=a)\nabla_{z_1}u_1(s_1(a),s_2(b);a)=0.
\]

Player 2's condition is analogous:

\[
\sum_aP(\theta_1=a\mid\theta_2=b)\nabla_{z_2}u_2(s_2(b),s_1(a);b)=0.
\]

Stacking these conditions gives a nonlinear system in all type-contingent logits. Its zero is a Bayesian Nash equilibrium under the interior assumption.

## 5. Numerical diagnostics

For each solver report the residual (\|F(x)\|_2), iteration count, initial-point sensitivity, covariance perturbation sensitivity, and out-of-sample performance. A trust-region step uses the ratio

\[
\rho_k=\frac{\phi(x_k)-\phi(x_k+p_k)}{-\nabla\phi(x_k)^Tp_k-\frac12p_k^T B_kp_k},
\]

to accept or reject the step and update the trust-region radius.
