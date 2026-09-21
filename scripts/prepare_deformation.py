#!/usr/bin/env python3
"""Construct the paper's paired deformation through Taylor order two.

Only small section spaces and multiplication tensors are constructed here.
The large Koszul operator is applied matrix-free by the Rust/CUDA programs.
Coefficients are ordinary coefficients of t**k, not kth derivatives.
The generator also supports genus 12 as a small independent test case.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np

P = 109
G, R, N, M, D = 12, 6, 9, 6, 22


def require(condition, message):
    if not condition:
        raise ValueError(message)


def zero(a):
    return not np.any(np.asarray(a) % P)


def rref(a):
    a = np.array(a, dtype=np.int64, copy=True) % P
    pivots, row = [], 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        a[[row, pivot]] = a[[pivot, row]]
        a[row] = a[row] * pow(int(a[row, col]), -1, P) % P
        factors = a[:, col].copy()
        factors[row] = 0
        a = (a - factors[:, None] * a[row]) % P
        pivots.append(col)
        row += 1
        if row == a.shape[0]:
            break
    return a, pivots


def nullspace(a):
    reduced, pivots = rref(a)
    free = [i for i in range(reduced.shape[1]) if i not in pivots]
    basis = np.zeros((len(free), reduced.shape[1]), dtype=np.int64)
    for j, col in enumerate(free):
        basis[j, col] = 1
        basis[j, pivots] = -reduced[:len(pivots), col] % P
    return basis


def inverse_matrix(a):
    size = len(a)
    reduced, pivots = rref(np.concatenate([a, np.eye(size, dtype=np.int64)], axis=1))
    require(pivots == list(range(size)), "singular basis change")
    return reduced[:, size:]


def jet_inverse(x):
    a, b, c = map(int, x)
    ai = pow(a, -1, P)
    return [ai, -b * ai**2 % P, (b*b*ai**3 - c*ai**2) % P]


def jet_product(a, b):
    return [sum(int(a[j])*int(b[k-j]) for j in range(k+1)) % P for k in range(3)]


def jet_matmul(a, b):
    return np.array([sum(a[j] @ b[k-j] for j in range(k+1)) % P for k in range(3)])


def jet_rref(a):
    a = np.array(a, dtype=np.int64, copy=True) % P
    pivots, row = [], 0
    for col in range(a.shape[2]):
        candidates = np.flatnonzero(a[0, row:, col])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        a[:, [row, pivot]] = a[:, [pivot, row]]
        inv = jet_inverse(a[:, row, col])
        old = a[:, row].copy()
        for k in range(3):
            a[k, row] = sum(old[j] * inv[k-j] for j in range(k+1)) % P
        factors = a[:, :, col].copy()
        factors[:, row] = 0
        # Descending order retains all lower-order coefficients until used.
        for k in range(2, -1, -1):
            a[k] = (a[k] - sum(factors[j, :, None] * a[k-j, row] for j in range(k+1))) % P
        pivots.append(col)
        row += 1
        if row == a.shape[1]:
            break
    return a, pivots


def jet_nullspace(a):
    reduced, pivots = jet_rref(a)
    require(zero(reduced[:, len(pivots):]), "constraint rank does not stay constant in this chart")
    free = [i for i in range(a.shape[2]) if i not in pivots]
    basis = np.zeros((3, len(free), a.shape[2]), dtype=np.int64)
    for j, col in enumerate(free):
        basis[0, j, col] = 1
        basis[:, j, pivots] = -reduced[:, :len(pivots), col] % P
    require(zero(jet_matmul(a, basis.transpose(0, 2, 1))), "lifted section constraints failed")
    return basis


def polynomial_product(a, b):
    return np.array([sum(np.convolve(a[j], b[k-j]) for j in range(k+1)) % P for k in range(3)])


def powers(x, velocity, degree):
    return np.array([[math.comb(i, k) * pow(x, i-k, P) * pow(velocity, k, P) % P
                      if i >= k else 0 for i in range(degree+1)] for k in range(3)], dtype=np.int64)


def evaluate(a, x, velocity):
    px = powers(x, velocity, a.shape[1]-1)
    return [sum(int(a[j] @ px[k-j]) for j in range(k+1)) % P for k in range(3)]


def constraints(points, velocities, multipliers, degree):
    result = np.zeros((3, len(points), degree+1), dtype=np.int64)
    for i, ((x, y), (vx, vy), glue) in enumerate(zip(points, velocities, multipliers)):
        xp, yp = powers(x, vx, degree), powers(y, vy, degree)
        for k in range(3):
            result[k, i] = (xp[k] - sum(glue[j] * yp[k-j] for j in range(k+1))) % P
    return result


def character_basis(matrix, shift):
    basis, weights = [], []
    for char in range(R):
        indices = [i for i in range(matrix.shape[1]) if (i+shift) % R == char]
        for vector in nullspace(matrix[:, indices]):
            full = np.zeros(matrix.shape[1], dtype=np.int64)
            full[indices] = vector
            basis.append(full)
            weights.append(char)
    return np.array(basis), weights


def homogeneous_lift(basis, constraint, shift):
    desired, weights = character_basis(constraint[0], shift)
    _, pivots = rref(basis[0])
    require(len(pivots) == len(basis[0]) == len(desired), "section basis dimension mismatch")
    change = desired[:, pivots] @ inverse_matrix(basis[0][:, pivots]) % P
    result = np.array([change @ b % P for b in basis])
    require(zero(result[0] - desired), "homogeneous basis change failed")
    require(zero(jet_matmul(constraint, result.transpose(0, 2, 1))), "homogeneous lift failed")
    return result, weights


def trim(a):
    a = [int(x) % P for x in a]
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def remainder(a, b):
    a, b = trim(a), trim(b)
    inv = pow(b[-1], -1, P)
    while a != [0] and len(a) >= len(b):
        offset, factor = len(a)-len(b), a[-1]*inv % P
        for j, x in enumerate(b):
            a[offset+j] = (a[offset+j] - factor*x) % P
        a = trim(a)
    return a


def projectively_coprime(a, b):
    if a[-1] == 0 and b[-1] == 0:
        return False
    a, b = trim(a), trim(b)
    while b != [0]:
        a, b = b, remainder(a, b)
    return len(a) == 1


def choose_pencil(w, weights):
    iu = [j for j, c in enumerate(weights) if c == 1]
    iv = [j for j, c in enumerate(weights) if c == R-1]
    require(len(iu) == len(iv) == 2, "unexpected pencil eigenspaces")
    for a, b in itertools.product(range(min(P, 8)), repeat=2):
        u = (w[:, iu[0]] + a*w[:, iu[1]]) % P
        v = (w[:, iv[0]] + b*w[:, iv[1]]) % P
        if projectively_coprime(u[0], v[0]):
            keep = [j for j in range(G-1) if j not in (iu[0], iv[0])]
            return u, v, w[:, keep], [weights[j] for j in keep], [a, b]
    raise ValueError("no regular pencil found in the bounded search")


def build_small(genus, prime):
    global P, G, R, N, M, D
    P, G = prime, genus
    require(G >= 6 and G % 2 == 0, "genus must be even and at least six")
    R, N, M, D = G//2, G-3, G//2, 2*G-2
    require(47 <= P <= 65521 and all(P % q for q in range(2, math.isqrt(P)+1)), "prime must be between 47 and 65521")
    require((P-1) % R == 0, "prime does not admit the required roots of unity")
    require(R % 2 == 0, "paired paper deformation requires even cyclic order")
    zeta = next(z for z in range(2, P) if pow(z, R, P) == 1 and all(pow(z, j, P) != 1 for j in range(1, R)))
    points = [(a*pow(zeta, j, P) % P, b*pow(zeta, j, P) % P)
              for a, b in [(1, 2), (4, 10)] for j in range(R)]
    require(len({x for pair in points for x in pair}) == 2*G, "endpoint collision")
    velocities = [(0, 0)] * G
    velocities[:R] = [(pow(zeta, 2*j, P), 1) for j in range(R)]

    factors = [np.array([[x*y, -x-y, 1], [vx*y+x*vy, -vx-vy, 0], [vx*vy, 0, 0]], dtype=np.int64) % P
               for (x, y), (vx, vy) in zip(points, velocities)]
    omega = []
    for j in range(G):
        form = np.array([[1], [0], [0]], dtype=np.int64)
        for i, factor in enumerate(factors):
            if i != j:
                form = polynomial_product(form, factor)
        omega.append(form)
    canonical_glue = []
    for form, (x, y), (vx, vy) in zip(omega, points, velocities):
        canonical_glue.append(jet_product(evaluate(form, x, vx), jet_inverse(evaluate(form, y, vy))))
    prym_glue = [[-v % P for v in b] for b in canonical_glue]
    product_glue = [jet_product(b, a) for b, a in zip(canonical_glue, prym_glue)]
    cw = constraints(points, velocities, prym_glue, D)
    co = constraints(points, velocities, canonical_glue, D)
    cm = constraints(points, velocities, product_glue, 2*D)
    w, ww = homogeneous_lift(jet_nullspace(cw), cw, 1)
    omega = np.array(omega).transpose(1, 0, 2)
    omega, ow = homogeneous_lift(omega, co, 1)
    sections = jet_nullspace(cm)
    require(w.shape[1] == G-1 and omega.shape[1] == G and sections.shape[1] == 3*G-3, "wrong section-space dimensions")
    u, v, vbasis, vw, pencil_combo = choose_pencil(w, ww)

    denominator = np.array([polynomial_product(q, omega[:, j]) for q in (u, v) for j in range(G)]).transpose(1, 0, 2)
    require(zero(jet_matmul(cm, denominator.transpose(0, 2, 1))), "denominator not in H0(omega L)")
    den_echelon, den_pivots = jet_rref(denominator)
    require(len(den_pivots) == 2*G, "pencil denominator has wrong rank")

    def reduce_mod_pencil(forms):
        return (forms - jet_matmul(forms[:, :, den_pivots], den_echelon)) % P

    # Construct quotient coordinates from the FULL H0(omega L).
    reduced_sections = reduce_mod_pencil(sections)
    quotient_basis, quotient_pivots = jet_rref(reduced_sections)
    require(len(quotient_pivots) == N and zero(quotient_basis[:, N:]), "Artinian quotient has wrong rank")
    quotient_basis = quotient_basis[:, :N]
    aw = [(j+2) % R for j in quotient_pivots]
    for basis, char in zip(quotient_basis[0], aw):
        require(all((i+2) % R == char for i in np.flatnonzero(basis)), "quotient basis is not homogeneous at t=0")
    products = np.array([polynomial_product(vbasis[:, i], omega[:, j]) for i in range(N) for j in range(G)]).transpose(1, 0, 2)
    require(zero(jet_matmul(cm, products.transpose(0, 2, 1))), "multiplication violates section constraints")
    reduced_products = reduce_mod_pencil(products)
    coordinates = reduced_products[:, :, quotient_pivots]
    require(zero(reduced_products - jet_matmul(coordinates, quotient_basis)), "quotient product reconstruction failed")
    mu = coordinates.reshape(3, N, G, N).transpose(0, 1, 3, 2)

    return mu, vw, ow, aw, zeta, points, velocities, pencil_combo, u, v, vbasis, omega, quotient_basis


def allowed_charges():
    """Central-fiber character shifts of orders zero, one and two."""
    return [[0], [1, R-1], sorted({0, 2 % R, (-2) % R})]


def check_charges(jets, charge_array, allowed, name):
    for k, charges in enumerate(allowed):
        legal = np.isin(charge_array % R, charges)
        require(zero(jets[k][~legal]), f"unexpected character charge in {name}_{k}")


def inverse_jet_matrix(a):
    result = np.zeros_like(a)
    result[0] = inverse_matrix(a[0])
    for k in (1, 2):
        result[k] = -result[0] @ sum(a[j] @ result[k-j] for j in range(1, k+1)) % P
    identity = np.zeros_like(a)
    identity[0] = np.eye(len(a[0]), dtype=np.int64)
    require(zero(jet_matmul(a, result)-identity), "formal matrix inverse failed")
    return result


def exterior_counts(weights, degree):
    counts = np.zeros((degree+1, R), dtype=np.int64)
    counts[0, 0] = 1
    for i, char in enumerate(weights):
        for q in range(min(i+1, degree), 0, -1):
            counts[q] += np.roll(counts[q-1], char)
    return counts[degree]


def tensor_counts(weights, degree, coefficient_weights):
    counts = exterior_counts(weights, degree)
    return sum(np.roll(counts, char) for char in coefficient_weights)


def build_instance(genus, prime):
    mu, vw, ow, aw, zeta, points, velocities, combo, u, v, vb, om, ab = build_small(genus, prime)
    w = next((i for i, char in enumerate(vw) if char == 2 and len(rref(mu[0, i])[1]) == N), None)
    require(w is not None, "no surjective basis direction of character two")
    order = [w] + [i for i in range(N) if i != w]
    mu, vb, vw = mu[:, order], vb[:, order], [vw[i] for i in order]
    mw = mu[:, 0]
    _, pivot_columns = rref(mw[0])
    free_columns = [i for i in range(G) if i not in pivot_columns]
    inv = inverse_jet_matrix(mw[:, :, pivot_columns])
    right_inverse = np.zeros((3, G, N), dtype=np.int64)
    right_inverse[:, pivot_columns, :] = inv
    kernel = np.zeros((3, G, 3), dtype=np.int64)
    kernel[0, free_columns, :] = np.eye(3, dtype=np.int64)
    kernel[:, pivot_columns, :] = -jet_matmul(inv, mw[:, :, free_columns]) % P
    kernel_weights = [ow[i] for i in free_columns]
    identity = np.zeros((3, N, N), dtype=np.int64)
    identity[0] = np.eye(N, dtype=np.int64)
    require(zero(jet_matmul(mw, right_inverse)-identity), "mu_w R != I")
    require(zero(jet_matmul(mw, kernel)), "mu_w B != 0")
    require(len(rref(kernel[0])[1]) == 3, "elimination kernel dimension is wrong")
    charges = allowed_charges()
    check_charges(mu, np.array(aw)[None, :, None] - np.array(vw)[:, None, None] - np.array(ow)[None, None, :], charges, "mu")
    check_charges(right_inverse, np.array(ow)[:, None] - np.array(aw)[None, :] + vw[0], charges, "R")
    check_charges(kernel, np.array(ow)[:, None] - np.array(kernel_weights)[None, :], charges, "B")
    ordinary_cols = tensor_counts(vw, M, ow)
    ordinary_rows = tensor_counts(vw, M-1, aw)
    ordinary_nnz = np.zeros(R, dtype=np.int64)
    for i, wi in enumerate(vw):
        subsets = exterior_counts(vw[:i]+vw[i+1:], M-1)
        for beta, a in np.argwhere(mu[0, i] != 0):
            ordinary_nnz += np.roll(subsets, wi+ow[int(a)])
    uweights = vw[1:]
    x_counts = tensor_counts(uweights, M, ow)
    z_counts = np.roll(tensor_counts(uweights, M-1, kernel_weights), vw[0])
    target_counts = np.roll(tensor_counts(uweights, M-2, aw), vw[0])
    eliminated_cols = x_counts + z_counts
    expected = {12: (106, 84), 20: (21886, 19448), 24: (323302, 293930)}
    if G in expected:
        require((int(target_counts[0]), int(eliminated_cols[0])) == expected[G], "unexpected eliminated sector-zero dimensions")
    data = {
        "fixture_format": "prym-cyclic-deformation-v1", "genus": G, "modulus": P,
        "status": {"small_tensor_verified": True, "koszul_rank_verified": False, "quadratic_certificate_verified": False},
        "cyclic": {"order": R, "zeta": zeta, "node_orbit_representatives": [[1, 2], [4, 10]], "points": points, "eta_signs": [-1]*G},
        "weights": {"V": vw, "A0": ow, "A1": aw},
        "pencil": {"u": u[0].tolist(), "v": v[0].tolist()},
        "artinian": {"mu": {"n": N, "g": G, "values": mu[0].reshape(-1).tolist()}},
        "koszul": {"rows": int(sum(ordinary_rows)), "columns": int(sum(ordinary_cols)), "sector_rows": ordinary_rows.tolist(), "sector_columns": ordinary_cols.tolist(), "sector_nnz": ordinary_nnz.tolist()},
        "third_section_elimination": {"w_V_index": 0, "w_weight": vw[0], "right_inverse_A0_by_A1": right_inverse[0].tolist(), "kernel_basis_A0_by_3": kernel[0].tolist(), "kernel_weights": kernel_weights, "sector_columns": eliminated_cols.tolist(), "sector_rows": target_counts.tolist()},
        "deformation": {"order": 2, "mode": "paired", "coefficient_convention": "coefficient of t^k, not kth derivative", "point_velocities": velocities, "allowed_charges": charges,
            "pencil_combo": combo, "pencil_u_coefficients": u.tolist(), "pencil_v_coefficients": v.tolist(),
            "mu_shape": [N, N, G], "mu_layout": "((i*n)+beta)*g+a", "mu_coefficients": mu.reshape(3, -1).tolist(),
            "right_inverse_shape": [G, N], "right_inverse_coefficients": right_inverse.reshape(3, -1).tolist(),
            "kernel_basis_shape": [G, 3], "kernel_basis_coefficients": kernel.reshape(3, -1).tolist(),
            "elimination_pivot_columns": pivot_columns, "elimination_free_columns": free_columns,
            "V_basis_coefficients": vb.tolist(), "A0_basis_coefficients": om.tolist(), "A1_basis_coefficients": ab.tolist(),
            "x_sector_columns": x_counts.tolist(), "z_sector_columns": z_counts.tolist(),
            "mu_nnz_by_order": [int(np.count_nonzero(a)) for a in mu],
            "max_mu_column_nnz_by_order": [int(np.count_nonzero(a, axis=1).max()) for a in mu],
            "verified": ["all section constraints through t^2", "regular pencil at t=0 including infinity", "full target section-space quotient", "product reconstruction through t^2", "mu_w R=I through t^2", "mu_w B=0 through t^2", "all Taylor coefficient character charges"]}
    }
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genus", type=int, choices=[12, 20, 24, 28], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prime = {12: 109, 20: 661, 24: 1009, 28: 1009}[args.genus]
    data = build_instance(args.genus, prime)
    data["deformation"]["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, separators=(",", ":"))+"\n")
    print(f"Verified paired Taylor input for genus {args.genus}: {args.out}")


if __name__ == "__main__":
    main()
