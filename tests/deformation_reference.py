#!/usr/bin/env python3
"""Exact small CPU reference for the eliminated quadratic deformation route.

Requires NumPy and SciPy.  This deliberately uses dense elimination only on
small character blocks: it is a reference for black-box implementations,
not the proposed genus-20/24 rank algorithm.  All jet coefficients are t^k
coefficients (there are no factorials).
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse

if not __debug__:
    raise RuntimeError('Run this exact certificate checker without Python -O; its verification assertions must execute.')


def rref(matrix, p):
    a = np.array(matrix, dtype=np.int64, copy=True) % p
    pivots = []
    row = 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        selected = row + int(candidates[0])
        a[[row, selected]] = a[[selected, row]]
        a[row] = a[row] * pow(int(a[row, col]), -1, p) % p
        factors = a[:, col].copy()
        factors[row] = 0
        a = (a - factors[:, None] * a[row]) % p
        pivots.append(col)
        row += 1
        if row == a.shape[0]:
            break
    return a, pivots


def rank(matrix, p):
    return len(rref(matrix, p)[1])


def kernel(matrix, p):
    a, pivots = rref(matrix, p)
    free = [j for j in range(a.shape[1]) if j not in set(pivots)]
    k = np.zeros((a.shape[1], len(free)), dtype=np.int64)
    k[free, np.arange(len(free))] = 1
    if pivots:
        k[pivots] = -a[:len(pivots), free] % p
    assert not np.any(np.asarray(matrix) @ k % p)
    return k


def inverse(matrix, p):
    n = matrix.shape[0]
    a, pivots = rref(np.hstack((matrix, np.eye(n, dtype=np.int64))), p)
    assert pivots[:n] == list(range(n)), 'singular normalization matrix'
    assert np.array_equal(a[:, :n], np.eye(n, dtype=np.int64))
    return a[:, n:]


def sparse_mod(a, p):
    a = a.tocsr()
    a.sum_duplicates()
    a.data %= p
    a.eliminate_zeros()
    return a


def differential(mu, source_subsets, target_subsets, g, n, p):
    target_index = {s: i for i, s in enumerate(target_subsets)}
    rr, cc, vv = [], [], []
    for column, subset in enumerate(source_subsets):
        for position, v_index in enumerate(subset):
            target = subset[:position] + subset[position + 1:]
            row = target_index[target]
            beta, alpha = np.nonzero(mu[v_index])
            rr.extend((row * n + beta).tolist())
            cc.extend((column * g + alpha).tolist())
            vv.extend((((-1) ** position) * mu[v_index, beta, alpha] % p).tolist())
    return sparse.coo_matrix((vv, (rr, cc)),
                             shape=(len(target_subsets) * n,
                                    len(source_subsets) * g),
                             dtype=np.int64).tocsr()


def assemble(fixture):
    g, p = fixture['genus'], fixture['modulus']
    if g > 14:
        raise ValueError('This exact dense-block reference is restricted to g <= 14.')
    if g % 2 or p > 100000:
        raise ValueError('Reference requires even genus and p <= 100000 (int64 bound).')
    n, m = g - 3, g // 2
    r = fixture['cyclic']['order']
    d = fixture['deformation']
    e = fixture['third_section_elimination']
    assert e['w_V_index'] == 0
    assert d['coefficient_convention'] == 'coefficient of t^k, not kth derivative'
    mu = np.asarray(d['mu_coefficients'], dtype=np.int64).reshape(3, n, n, g)
    right = np.asarray(d['right_inverse_coefficients'], dtype=np.int64).reshape(3, g, n)
    basis = np.asarray(d['kernel_basis_coefficients'], dtype=np.int64).reshape(3, g, 3)
    for k in range(3):
        mr = sum(mu[a, 0] @ right[k-a] for a in range(k+1)) % p
        mb = sum(mu[a, 0] @ basis[k-a] for a in range(k+1)) % p
        assert np.array_equal(mr, np.eye(n, dtype=np.int64) if k == 0 else np.zeros((n,n), dtype=np.int64))
        assert not np.any(mb)

    subsets = {k: list(itertools.combinations(range(1, n), k))
               for k in (m, m-1, m-2)}
    dm = [differential(mu[k], subsets[m], subsets[m-1], g, n, p) for k in range(3)]
    d2 = [differential(mu[k], subsets[m-1], subsets[m-2], g, n, p) for k in range(3)]
    identity = sparse.eye(len(subsets[m-1]), dtype=np.int64, format='csr')
    rb = [sparse.kron(identity, sparse.csr_matrix(right[k]), format='csr') for k in range(3)]
    bb = [sparse.kron(identity, sparse.csr_matrix(basis[k]), format='csr') for k in range(3)]
    matrices = []
    for k in range(3):
        x = sparse.csr_matrix((d2[0].shape[0], dm[0].shape[1]), dtype=np.int64)
        z = sparse.csr_matrix((d2[0].shape[0], bb[0].shape[1]), dtype=np.int64)
        for a in range(k+1):
            z = sparse_mod(z + sparse_mod(d2[a] @ bb[k-a], p), p)
            for b in range(k-a+1):
                x = sparse_mod(x - sparse_mod(sparse_mod(d2[a] @ rb[b], p) @ dm[k-a-b], p), p)
        matrices.append(sparse.hstack((x, z), format='csr'))

    weights = fixture['weights']
    vw = weights['V']
    x_weights = [sum(vw[i] for i in subset) + w
                 for subset in subsets[m] for w in weights['A0']]
    z_weights = [sum(vw[i] for i in subset) + w + e['w_weight']
                 for subset in subsets[m-1] for w in e['kernel_weights']]
    row_weights = [sum(vw[i] for i in subset) + w + e['w_weight']
                   for subset in subsets[m-2] for w in weights['A1']]
    columns = np.asarray(x_weights + z_weights) % r
    rows = np.asarray(row_weights) % r
    for k, matrix in enumerate(matrices):
        coo = matrix.tocoo()
        charges = set(((rows[coo.row] - columns[coo.col]) % r).tolist())
        assert charges <= set(d['allowed_charges'][k]), (k, charges)
    assert [int(np.sum(columns == c)) for c in range(r)] == e['sector_columns']
    assert [int(np.sum(rows == c)) for c in range(r)] == e['sector_rows']

    # Independent factored action/transpose checks; do not materialize F in CUDA.
    rng = np.random.default_rng(271828)
    v = rng.integers(0, p, size=(len(columns), 2), dtype=np.int64)
    y = rng.integers(0, p, size=(len(rows), 2), dtype=np.int64)
    cut = len(x_weights)
    for k, matrix in enumerate(matrices):
        action = np.zeros((len(rows), 2), dtype=np.int64)
        transpose_x = np.zeros((cut, 2), dtype=np.int64)
        transpose_z = np.zeros((len(columns)-cut, 2), dtype=np.int64)
        for a in range(k+1):
            action = (action + d2[a] @ (bb[k-a] @ v[cut:] % p)) % p
            transpose_z = (transpose_z + bb[k-a].T @ (d2[a].T @ y % p)) % p
            for b in range(k-a+1):
                c = k-a-b
                action = (action - d2[a] @ (rb[b] @ (dm[c] @ v[:cut] % p) % p)) % p
                transpose_x = (transpose_x - dm[c].T @ (rb[b].T @ (d2[a].T @ y % p) % p)) % p
        assert np.array_equal(action, matrix @ v % p)
        assert np.array_equal(np.vstack((transpose_x, transpose_z)), matrix.T @ y % p)
    return matrices, rows, columns


def check(fixture, include_vectors=False):
    started = time.perf_counter()
    p, r = fixture['modulus'], fixture['cyclic']['order']
    matrices, row_weights, column_weights = assemble(fixture)
    f0, f1, f2 = matrices
    row_indices = [np.flatnonzero(row_weights == c) for c in range(r)]
    col_indices = [np.flatnonzero(column_weights == c) for c in range(r)]
    blocks = [f0[row_indices[c]][:, col_indices[c]].toarray() for c in range(r)]
    kernels = [kernel(a, p) for a in blocks]
    profiles = [{'sector': c, 'rows': int(a.shape[0]), 'columns': int(a.shape[1]),
                 'rank': int(a.shape[1] - kernels[c].shape[1]),
                 'nullity': int(kernels[c].shape[1])} for c, a in enumerate(blocks)]
    assert kernels[0].shape[1] == 2, 'Reference expects a two-dimensional kernel in sector zero.'
    assert all(k.shape[1] == 0 for k in kernels[1:]), 'Other base sectors must be injective.'
    k0 = kernels[0]
    k = np.zeros((f0.shape[1], 2), dtype=np.int64)
    k[col_indices[0]] = k0
    assert not np.any(f0 @ k % p)
    b = f1 @ k % p
    s = np.zeros_like(k)
    corrections = []
    for c in range(r):
        bc = b[row_indices[c]]
        if not np.any(bc):
            continue
        # Production recovery route: kernel of [A|-b], then normalize the
        # bottom two coordinates. No left nullspace or inverse of A is used.
        augmented = np.hstack((blocks[c], -bc % p))
        h = kernel(augmented, p)
        bottom = h[-2:]
        pivots = rref(bottom, p)[1]
        assert len(pivots) == 2, ('first-order obstruction', c)
        selected = h[:, pivots[:2]]
        normalized = selected @ inverse(selected[-2:], p) % p
        assert np.array_equal(normalized[-2:], np.eye(2, dtype=np.int64))
        sc = normalized[:-2]
        assert np.array_equal(blocks[c] @ sc % p, bc)
        s[col_indices[c]] = sc
        corrections.append({'sector': c, 'augmented_shape': list(augmented.shape),
                            'augmented_nullity': int(h.shape[1]),
                            'bottom_rank': 2, 'residual_zero': True})
    assert np.array_equal(f0 @ s % p, b)
    z = (f2 @ k - f1 @ s) % p
    z0 = z[row_indices[0]]
    j = rref(k0.T, p)[1]
    assert len(j) == 2
    complement = [i for i in range(blocks[0].shape[1]) if i not in j]
    completed = np.hstack((blocks[0][:, complement], z0))
    completed_rank = rank(completed, p)
    assert completed_rank == completed.shape[1], 'Quadratic obstruction is not injective.'
    result = {
        'genus': fixture['genus'], 'modulus': p, 'cyclic_order': r,
        'mode': fixture['deformation']['mode'],
        'basis_order': 'lexicographic U subsets; coefficient index fastest; x then z',
        'F_shape': list(f0.shape), 'F_nnz_by_order': [int(f.nnz) for f in matrices],
        'sector_profiles': profiles, 'original_kernel_dimension': 2,
        'first_order_corrections': corrections, 'first_order_residual_zero': True,
        'kernel_pivot_rows_sector0': j,
        'completion_shape': list(completed.shape), 'completion_rank': completed_rank,
        'quadratic_obstruction_rank': 2,
        'all_jet_charges_verified': True, 'factored_actions_and_transposes_verified': True,
        'generic_injectivity_verified': True,
        'elapsed_seconds': round(time.perf_counter()-started, 6),
    }
    if include_vectors:
        result['certificate'] = {
            'K0': k0.tolist(),
            'S_by_sector': {str(c['sector']): s[col_indices[c['sector']]].tolist() for c in corrections},
            'Z0': z0.tolist(), 'J': j,
            'source_sector_indices': {str(c): col_indices[c].tolist() for c in range(r)},
            'target_sector_indices': {str(c): row_indices[c].tolist() for c in range(r)},
        }
    return result
