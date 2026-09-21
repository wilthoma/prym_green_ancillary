#!/usr/bin/env python3
"""Independent exact checks of the six small paper inputs (standard library only).

The section/product calculation reconstructs the finite-field linear systems
from node coordinates. It does not call the Rust or CUDA implementations.
Large Koszul ranks and saved deformation obstruction vectors are not checked
by this script; those require the computational pipeline.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def rref(rows, prime):
    """Return reduced rows and pivot columns over the prime field."""
    a = [[x % prime for x in row] for row in rows]
    rank, pivots = 0, []
    for column in range(len(a[0])):
        pivot = next((i for i in range(rank, len(a)) if a[i][column]), None)
        if pivot is None:
            continue
        a[rank], a[pivot] = a[pivot], a[rank]
        inverse = pow(a[rank][column], -1, prime)
        a[rank] = [x * inverse % prime for x in a[rank]]
        for i in range(len(a)):
            if i != rank and a[i][column]:
                factor = a[i][column]
                a[i] = [(x - factor * y) % prime for x, y in zip(a[i], a[rank])]
        pivots.append(column)
        rank += 1
        if rank == len(a):
            break
    return a, pivots


def kernel(rows, prime):
    reduced, pivots = rref(rows, prime)
    basis = []
    for free in range(len(rows[0])):
        if free in pivots:
            continue
        vector = [0] * len(rows[0])
        vector[free] = 1
        for i, pivot in enumerate(pivots):
            vector[pivot] = -reduced[i][free] % prime
        basis.append(vector)
    require(all(sum(x*y for x, y in zip(row, v)) % prime == 0
                for row in rows for v in basis), "nullspace residual")
    return basis, len(pivots)


def convolution(a, b, prime):
    result = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i+j] = (result[i+j] + x*y) % prime
    return result


def trim(poly):
    poly = poly[:]
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def gcdpoly(a, b, prime):
    a, b = trim(a), trim(b)
    while b != [0]:
        remainder = a[:]
        inverse = pow(b[-1], -1, prime)
        while remainder != [0] and len(remainder) >= len(b):
            shift, factor = len(remainder)-len(b), remainder[-1]*inverse % prime
            for j, value in enumerate(b):
                remainder[shift+j] = (remainder[shift+j] - factor*value) % prime
            remainder = trim(remainder)
        a, b = b, remainder
    inverse = pow(a[-1], -1, prime)
    return [x*inverse % prime for x in a]


def constraints(pairs, degree, prime, power):
    endpoints = [x for pair in pairs for x in pair]
    derivatives = {x: math.prod((x-y) % prime for y in endpoints if y != x) % prime
                   for x in endpoints}
    result = []
    for x, y in pairs:
        ratio = pow(derivatives[x] * pow(derivatives[y], -1, prime) % prime, power, prime)
        result.append([(pow(x, j, prime)-ratio*pow(y, j, prime)) % prime
                       for j in range(degree+1)])
    return result


def section_checks(genus, prime, pairs):
    """Reconstruct H0(L), H0(L^2), and Sym^2 H0(L) -> H0(L^2)."""
    source = constraints(pairs, 2*genus-2, prime, 1)
    target = constraints(pairs, 4*genus-4, prime, 2)
    basis, source_rank = kernel(source, prime)
    require(len(basis) == genus-1, "section-space dimension")
    products = [convolution(a, b, prime)
                for i, a in enumerate(basis) for b in basis[i:]]
    require(all(sum(x*y for x, y in zip(row, product)) % prime == 0
                for row in target for product in products), "product gluing residual")
    product_rank = len(rref(products, prime)[1])
    target_dimension = 4*genus-3-len(rref(target, prime)[1])
    require(product_rank == target_dimension == 3*genus-3, "quadratic multiplication not surjective")
    common = basis[0]
    for b in basis[1:]:
        common = gcdpoly(common, b, prime)
    require(len(common) == 1 and any(b[-1] for b in basis), "section basepoint")
    return {"section_dimension": len(basis), "constraints_rank": source_rank,
            "quadratic_product_rank": product_rank, "target_dimension": target_dimension,
            "basepoint_free": True}


def character(poly, order, prime):
    return {(j+1) % order for j, value in enumerate(poly) if value % prime}


def check_inverse_jets(instance):
    """Check mu_w R=I and mu_w B=0, including t,t^2 when supplied."""
    genus, prime = instance["genus"], instance["modulus"]
    n = genus-3
    elimination = instance["third_section_elimination"]
    w_index = elimination["w_V_index"]
    deformation = instance.get("deformation")
    if deformation:
        mus = deformation["mu_coefficients"]
        inverses = deformation["right_inverse_coefficients"]
        kernels = deformation["kernel_basis_coefficients"]
        require(len(mus) == len(inverses) == len(kernels) == 3, "Taylor order")
    else:
        mus = [instance["artinian"]["mu"]["values"]]
        inverses = [[x for row in elimination["right_inverse_A0_by_A1"] for x in row]]
        kernels = [[x for row in elimination["kernel_basis_A0_by_3"] for x in row]]
    for order in range(len(mus)):
        require(len(mus[order]) == n*n*genus and len(inverses[order]) == genus*n
                and len(kernels[order]) == genus*3, "Taylor array dimensions")
        for row in range(n):
            for column in range(n):
                value = sum(mus[k][(w_index*n+row)*genus+a] * inverses[order-k][a*n+column]
                            for k in range(order+1) for a in range(genus)) % prime
                require(value == int(order == 0 and row == column), "mu_w R != I")
            for column in range(3):
                value = sum(mus[k][(w_index*n+row)*genus+a] * kernels[order-k][a*3+column]
                            for k in range(order+1) for a in range(genus)) % prime
                require(value == 0, "mu_w B != 0")
    kernel_rows = [kernels[0][i*3:(i+1)*3] for i in range(genus)]
    require(len(rref(kernel_rows, prime)[1]) == 3, "mu_w kernel basis has wrong rank")


def check_input(case, instance, *, products=True):
    genus, prime, order, zeta = (case[k] for k in ("genus", "prime", "order", "zeta"))
    require(genus in (20, 22, 24, 26, 28, 30) and order == genus//2, "paper genus/order")
    require(prime > 2 and all(prime % d for d in range(2, math.isqrt(prime)+1)), "not prime")
    require(pow(zeta, order, prime) == 1 and all(pow(zeta, j, prime) != 1
            for j in range(1, order)), "root of unity has wrong order")
    require((instance["genus"], instance["modulus"]) == (genus, prime), "input parameters")
    cyclic = instance["cyclic"]
    require((cyclic["order"], cyclic["zeta"]) == (order, zeta), "cyclic parameters")
    require(cyclic["node_orbit_representatives"] == [[1, 2], [4, 10]], "orbit representatives")
    pairs = [[a*pow(zeta, j, prime) % prime, b*pow(zeta, j, prime) % prime]
             for a, b in [(1, 2), (4, 10)] for j in range(order)]
    require(cyclic["points"] == pairs, "point recipe")
    require(len({x for pair in pairs for x in pair}) == 2*genus, "endpoint collision")
    require(cyclic["eta_signs"] == [-1]*genus, "Prym signs")
    rows, columns = (genus-3)*math.comb(genus-3, order-1), genus*math.comb(genus-3, order)
    require(instance["koszul"]["rows"] == case["ordinary_rows"] == rows, "ordinary rows")
    require(instance["koszul"]["columns"] == case["ordinary_columns"] == columns, "ordinary columns")
    elimination = instance["third_section_elimination"]
    for key in ("sector_rows", "sector_columns"):
        require(elimination[key] == case[key] and len(case[key]) == order, key)
    u, v = instance["pencil"]["u"], instance["pencil"]["v"]
    require(character(u, order, prime) == {1} and character(v, order, prime) == {order-1}, "pencil characters")
    require(len(gcdpoly(u, v, prime)) == 1 and (u[-1] or v[-1]), "pencil common projective zero")
    deformation = instance.get("deformation")
    w = elimination.get("w_polynomial")
    if w is None:
        w = deformation["V_basis_coefficients"][0][elimination["w_V_index"]]
    require(elimination["w_weight"] == 2 and character(w, order, prime) == {2}, "w character")
    if case["deformation"] == "paired":
        require(deformation is not None and deformation["mode"] == "paired", "deformation mode")
        require(deformation["order"] == 2 and deformation["coefficient_convention"] ==
                "coefficient of t^k, not kth derivative", "Taylor convention")
        velocities = [[pow(zeta, 2*j, prime), 1] for j in range(order)] + [[0, 0]]*order
        require(deformation["point_velocities"] == velocities, "paired velocities")
        require(deformation["pencil_combo"] == [0, 1], "deformation pencil choice")
        require(deformation["pencil_u_coefficients"][0] == u and
                deformation["pencil_v_coefficients"][0] == v, "pencil constant coefficients")
        require(deformation["mu_coefficients"][0] == instance["artinian"]["mu"]["values"], "mu constant coefficients")
    else:
        require(deformation is None, "unexpected deformation")
    check_inverse_jets(instance)
    result = {"genus": genus, "prime": prime, "input_and_small_elimination_verified": True}
    if products:
        result.update(section_checks(genus, prime, pairs))
    return result


def load_cases():
    manifest = json.loads((ROOT/"data/cases.json").read_text())
    require(manifest["schema_version"] == 1, "case schema version")
    require([c["genus"] for c in manifest["cases"]] == [20, 22, 24, 26, 28, 30], "paper case coverage")
    require(manifest["settings"] == {"vector_count": 4, "numerical_seed": 1,
            "rowmix_rounds": 2, "rowmix_seed": 10001,
            "rowmix_algorithm": "paired-sl2-target-v1"}, "paper numerical settings")
    return manifest["cases"]


def load_input(case, data_dir=None):
    raw = ((Path(data_dir) if data_dir is not None else ROOT/"data")/case["input"]).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == case["sha256"], f"g={case['genus']}: input SHA256")
    return json.loads(raw)


def check(genus, prime, zeta):
    """Reconstruct the paper point recipe and check small section geometry."""
    order = genus//2
    require(genus in (20, 22, 24, 26, 28, 30), "paper genus")
    require(prime > 2 and all(prime % d for d in range(2, math.isqrt(prime)+1)), "not prime")
    require(pow(zeta, order, prime) == 1 and all(pow(zeta, j, prime) != 1
            for j in range(1, order)), "root of unity has wrong order")
    pairs = [[a*pow(zeta, j, prime) % prime, b*pow(zeta, j, prime) % prime]
             for a, b in [(1, 2), (4, 10)] for j in range(order)]
    require(len({x for pair in pairs for x in pair}) == 2*genus, "endpoint collision")
    return {"genus": genus, "prime": prime, **section_checks(genus, prime, pairs)}


def validate_input(case, data_dir=None):
    """Check frozen bytes, recorded small tensors and independent geometry."""
    return check_input(case, load_input(case, data_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the check results as JSON.")
    args = parser.parse_args()
    results = [validate_input(case) for case in load_cases()]
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(f"g={result['genus']}: input, geometry, multiplication and small elimination checks passed")
        print("Large Koszul ranks are checked by the CUDA reproduction pipeline.")


if __name__ == "__main__":
    main()
