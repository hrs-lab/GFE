#!/usr/bin/env python3
"""Reproducible formula checks for main_revised.tex.

These checks are not a PDE simulation, a spectral-gap test, or a substitute
for the analytic proofs. No random samples or network access are used.

Usage:
    python verify_revision.py --output verification_results.json
Dependencies:
    numpy, scipy, sympy
"""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any, Callable
import warnings

import numpy as np
import scipy
from scipy.integrate import IntegrationWarning, quad
import sympy as sp

EPSABS = 2e-12
EPSREL = 2e-12


def integrate(fun: Callable[[float], float], a: float, b: float,
              *, points: list[float] | None = None) -> tuple[float, float]:
    """One-dimensional adaptive quadrature; warnings are treated as errors."""
    opts: dict[str, Any] = dict(epsabs=EPSABS, epsrel=EPSREL, limit=300)
    if points is not None:
        opts['points'] = points
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        value, error = quad(fun, a, b, **opts)
    if not math.isfinite(value) or not math.isfinite(error):
        raise ArithmeticError('Non-finite quadrature result')
    return float(value), float(error)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def symbol_check() -> dict[str, Any]:
    s = sp.symbols('s')
    K = 6 / ((s + 1) * (s + 2))
    target = -(s - 2) * (s + 5) / ((s + 1) * (s + 2))
    discrepancy = sp.factor(2 * K - 1 - target)
    ratio = sp.cancel((2 - s) / target)
    decomposition = s - 2 + 12 / (s + 5)
    require(discrepancy == 0, 'Beta symbol identity failed')
    require(sp.cancel(ratio - decomposition) == 0, 'Flux multiplier failed')
    return {'symbol_identity_exact': True,
            'flux_multiplier_identity_exact': True,
            'critical_derivative': str(sp.diff(K, s).subs(s, 2))}


def negative_rate_check() -> dict[str, Any]:
    x = 0.9
    # Substitute z=1/y on [1,infinity); this avoids overflow in y**100.
    part1, error1 = integrate(lambda y: y**(-6) / (1 + y**100), x, 1.0)
    part2, error2 = integrate(lambda z: z**104 / (1 + z**100), 0.0, 1.0)
    value = (100 * x**100 / (1 + x**100)**2
             - 2 / (1 + x**100) + 12 * x**5 * (part1 + part2))
    bound = (100 * x**100 / (1 + x**100)**2 - 2 / (1 + x**100)
             + 12 * ((1 - x**5) / 5 + x**5 / 105))
    require(value <= bound + 1e-11, 'Numerical value exceeds analytic bound')
    require(bound < -0.94, 'Strict negative analytic bound failed')
    return {'x': x, 'flux_divided_by_C': value, 'analytic_upper_bound': bound,
            'quadrature_error_estimate': 12 * x**5 * (error1 + error2),
            'negative_without_projection': True}


def right_inverse_check() -> dict[str, Any]:
    x = sp.symbols('x', positive=True)
    psi = 12 * x / (1 + x)**5
    F = sp.factor(psi * (x / (1 + x) - sp.Rational(3, 5)))
    H = -12 * x**3 / (5 * (1 + x)**5)
    G = -12 * x * (4 * x - 1) / (5 * (1 + x)**6)
    require(sp.cancel(sp.diff(H, x) - x * F) == 0, 'Primitive identity failed')
    require(sp.cancel(G - (-F + 2 * H / x**2)) == 0, 'Right inverse formula failed')
    require(sp.integrate(x * F, (x, 0, sp.oo)) == 0, 'First moment failed')
    ff = sp.lambdify(x, F, 'numpy')
    gf = sp.lambdify(x, G, 'numpy')
    test_points = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    residuals = []
    for point in test_points:
        integral, _ = integrate(lambda y: float(gf(y)) / y, point, math.inf)
        residuals.append(abs(2 * integral - float(gf(point)) - float(ff(point))))
    require(max(residuals) < 1e-9, 'Quadrature right-inverse residual too large')
    return {'p': 4, 'first_moment_exactly_zero': True,
            'primitive_and_inverse_identities_exact': True,
            'test_points': test_points, 'absolute_residuals': residuals,
            'maximum_absolute_residual': max(residuals)}


def perturbation_check() -> dict[str, Any]:
    u = sp.symbols('u', real=True)
    a, b = sp.Rational(1, 4), sp.Rational(3, 4)
    H = (u - a)**4 * (b - u)**4
    h0 = sp.diff(H, u, 2)
    moment0 = sp.integrate(h0, (u, a, b))
    moment1 = sp.integrate(u * h0, (u, a, b))
    require(moment0 == 0 and moment1 == 0, 'Perturbation moments failed')
    # Solve for polynomial roots to split the absolute-value quadrature.
    roots = sp.nroots(h0, maxsteps=200)
    zeros = sorted(set(float(sp.re(z)) for z in roots
                       if abs(float(sp.im(z))) < 1e-10
                       and float(a) < float(sp.re(z)) < float(b)))
    critical = sp.nroots(sp.diff(h0, u), maxsteps=200)
    candidates = [float(a), float(b)] + [float(sp.re(z)) for z in critical
        if abs(float(sp.im(z))) < 1e-10 and float(a) < float(sp.re(z)) < float(b)]
    hf = sp.lambdify(u, h0, 'numpy')
    sup0 = max(abs(float(hf(z))) for z in candidates)
    h20, err = integrate(lambda z: abs(float(hf(z))) / z**2,
                        float(a), float(b), points=zeros)
    p = 4.0
    Cp = 1 + 2**(p + 2) * max(1 / 3, 1 / (p - 2))
    scale = min(0.25 / sup0, 0.20 / (2 * Cp * h20))
    sup_h = scale * sup0
    theta = 2 * Cp * scale * h20
    require(sup_h < 0.5 and theta < 0.25, 'Smallness hypotheses failed')
    return {'p': p, 'a': float(a), 'b': float(b), 'C_p': Cp,
            'zeroth_moment_exactly_zero': True,
            'first_moment_exactly_zero': True,
            'scale': scale, 'sup_norm_h': sup_h,
            'H_2_h': scale * h20, 'theta_h': theta,
            'relative_flux_error_bound': theta / (1 - theta),
            'relative_flux_lower_bound': (1 - 2 * theta) / (1 - theta),
            'relative_flux_upper_bound': 1 / (1 - theta),
            'rate_upper_bound': p / (1 - theta),
            'quadrature_error_estimate_H2': scale * err,
            'scope': 'Checks theorem hypotheses; does not numerically sum the flux series.'}


def matrix_check() -> dict[str, Any]:
    results = {}
    for M in [10.0, 100.0, 1000.0]:
        matrix = np.array([[-1.0, M], [0.0, -1.0]])
        condition = float(np.linalg.cond(matrix, 2))
        results[str(int(M))] = {'condition_2': condition,
                               'condition_2_divided_by_M_squared': condition / M**2}
    require(results['1000']['condition_2'] > 1e6, 'Ill-conditioning test failed')
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('verification_results.json'))
    args = parser.parse_args()
    record = {
        'scope': ('Deterministic formula checks only. No PDE eigenvalue solve, '
                  'spectral-gap claim, or convergence study.'),
        'software': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'sympy': sp.__version__},
        'quadrature': {'epsabs': EPSABS, 'epsrel': EPSREL},
        'beta_kernel': symbol_check(),
        'negative_rate': negative_rate_check(),
        'relative_right_inverse': right_inverse_check(),
        'feasible_kernel_perturbation': perturbation_check(),
        'triangular_conditioning': matrix_check(),
        'all_checks_passed': True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n',
                           encoding='utf-8')
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
