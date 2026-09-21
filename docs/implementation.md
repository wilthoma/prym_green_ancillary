# Mathematics and implementation

This repository implements the fixed computations in Sections 3–4 of the
paper. The formulas and coordinate conventions are retained from the authors'
working code; [THIRD_PARTY.md](../THIRD_PARTY.md) records the extraction sources.
[RESULTS.md](../RESULTS.md) describes the original runs. A new run computes its
own sequences, ranks, and residuals; it does not reuse those recorded results.

## From nodal points to small multiplication tensors

Write `g=2m`, `n=g-3`, and work in the prime field specified in
[data/cases.json](../data/cases.json). With zero-based `j=0,...,m-1`, the node
pairs are `(zeta^j,2*zeta^j)` and `(4*zeta^j,10*zeta^j)`. All Prym signs are
`-1`. The point order, bases, pencil, and elimination data are stored in the
six small JSON inputs, so they are part of the reproducible input.

Put `D(z)=product_j (z-P_j)(z-Q_j)`. A polynomial representing a canonical
section satisfies `f(P_j)=-D'(P_j)/D'(Q_j)*f(Q_j)`; for
`L=omega_C tensor eta` the sign is positive. The routines build these linear
constraints and solve them over the finite field. Coefficient index `e` means
`z^e`, equivalently `x0^(d-e)*x1^e` in a degree-`d` homogeneous binary form.
Canonical and Prym characters are `e+1 mod m`; product-section characters
are `e+2 mod m`.

The pencil sections `u,v` have characters `1,m-1`. Their affine gcd is constant
and their highest coefficients are not both zero, checking that they have no
common zero on the entire projective line. Set

```text
W  = H0(L),                             dim W  = g-1
V  = W / <u,v>,                         dim V  = n
A0 = H0(omega_C),                       dim A0 = g
A1 = H0(omega_C tensor L) / <u,v>A0,     dim A1 = n.
```

Multiplication by a basis section `t_i` of `V` gives
`mu_i:A0 -> A1`. The stored tensor has shape `(n,n,g)` and offset
`((i*n)+beta)*g+a`: the coefficient index `a` varies fastest. Polynomial
products are solved in the combined denominator/quotient basis, and exact
reconstruction residuals are checked.

For odd `m`, `cyclic::verify_cyclic_instance` reconstructs the tensor from the
stored polynomial bases, checks gluing and weights, and verifies elimination
identities and dimensions. It enumerates subsets for metadata but never
allocates the large Koszul matrix. For even `m`, the Python preparation code
rebuilds all Taylor data and compares them with the fixed input, ignoring only
the historical generator-source hash.

The separate `scripts/check_geometry.py` reconstructs `H0(L)` and `H0(L^2)`
directly from the nodes. It checks basepoint-freeness and surjectivity of
`Sym^2 H0(L) -> H0(L^2)`, with product rank `3g-3`, as used in the paper's
normal-generation argument. It also checks the fixed recipe and the small
right-inverse/kernel identities. These checks do not compute a large rank.

## Elimination and character sectors

The Koszul map uses lexicographic exterior subsets and the alternating sign
`(-1)^position` when an element is removed from an ordered subset. Choose
`w` of character `q=2` with surjective `mu_w`, and write `V=<w> direct_sum U`.
Let `R:A1 -> A0` be a right inverse, and `B:K_w -> A0` the inclusion of the
three-dimensional kernel of `mu_w`. The small-data checks verify
`mu_w R=I`, `mu_w B=0`, and independence of the columns of `B`.

For the remaining Koszul differentials `D_m,D_(m-1)`, eliminating the component
containing `w` gives exactly the paper's operator

```text
F(x,z) = D_(m-1) (B z - R D_m x),          ker F = ker Phi_g,
x in wedge^m U tensor A0,
z in wedge^(m-1) U tensor K_w.
```

Here `R` and `B` act separately on each exterior-subset coefficient block.
This is elimination of an equation, not a further quotient of the module.
CUDA applies the three factors and their transposes without forming `F`.
The source coordinates list all `x` coordinates first, then all `z`
coordinates; within each group, the coefficient index varies fastest.

Sector labels incorporate the omitted `w`: an `x` coordinate has the sum of
its subset weights plus its `A0` weight; a `z` coordinate has the subset sum
plus its `K_w` weight plus `q`. Target coordinates have the subset sum plus
their `A1` weight plus `q`. All labels are modulo `m`. This implements the
`c-q` convention in Section 3.5 of the paper, rather than silently relabeling
the second source summand or target.

## Paired deformation and the obstruction

For `g=20,24,28`, only the first orbit moves:
`P_j(t)=zeta^j+t*zeta^(2j)` and `Q_j(t)=2*zeta^j+t`.
`scripts/prepare_deformation.py` works modulo `t^3`, using pivots invertible
at `t=0` to continue the section bases, pencil, quotient coordinates, right
inverse, and kernel inclusion. It checks section constraints, product
reconstruction, and `mu_w(t)R(t)=I`, `mu_w(t)B(t)=0` through order two.

Every stored order-`k` array is a coefficient of `t^k`, not a kth derivative.
For example, `F^(2)` includes every product of factor coefficients whose
orders sum to two. Sector labels continue to refer to the special fiber;
the allowed character changes are `{0}`, `{1,-1}`, and `{0,2,-2}` at orders
zero, one, and two respectively.

First establish that sector zero has exactly a two-dimensional kernel:
combine the BCW lower bound `N-2` with two independent vectors `K` satisfying
`F^(0)_0 K=0`. Every other base sector must attain full column rank.
The first-order right-hand side `b=F^(1)K` lies in sectors `1,m-1`.
For each, compute a kernel of the augmented operator
`(x,c) -> F^(0)_c x - b_c c`, and normalize two columns so their bottom
two coordinates form the identity. The upper coordinates then give `S_code`
satisfying `F^(0) S_code=F^(1)K`.

The paper uses `S_paper=-S_code`. Thus the implementation's obstruction
`Z=F^(2)K-F^(1)S_code` is the paper's `F^(2)K+F^(1)S_paper`.
Project to sector zero, choose two rows `J` of `K` forming an invertible
`2 x 2` matrix, and replace the corresponding columns of `F^(0)_0` by `Z_0`.
The remaining columns span the original image. A lower bound `N` for this
replacement matrix proves independence of the two obstruction classes.
This excludes a formal kernel vector with nonzero constant coefficient,
and hence gives injectivity over `F_p((t))` as explained in Section 3.6.

Kernel recovery checks independence and residuals against the original
operator on CUDA. The Rust stages consume the newly generated reports;
without a matching report they recompute those residuals on CPU. They reject
missing or duplicate first-order correction sectors. A deficient lower bound
alone is never treated as proof of a kernel dimension.

## Wiedemann sequences and the rank lower bound

For any operator `A` in the pipeline, CUDA uses the symmetric square operator
`S = Dc A^T T^T Dr T A Dc`. The diagonal matrices `Dc,Dr` have nonzero entries;
`T` is the invertible target-row mixing transformation. The fixed recipe uses
block width `b=4`, seed `1`, and two `paired-sl2-target-v1` mixing rounds with
seed `10001`. Symmetry allows the Gram sequence to be stored triangularly.
The elementary inequality `rank(S) <= rank(A)` is sufficient; equality of
these ranks over a finite field is not assumed.

For an initial block `V`, WDM stores `V^T S^k V`, including `k=0`, together
with dimensions, modulus, diagonal preconditioners, initial vectors, and
current vectors. The `.wdm.zst` file is compressed text; only the upper
triangle of each symmetric `b x b` Gram matrix is stored. The neighboring
`.rowmix.json` records the mixing recipe and hashes needed for kernel recovery.
The CUDA writer and Rust reader must agree on these layouts.

`prepare_wdm_sequence_for_rank` removes the initial Gram term. The rank
algorithm therefore sees the shifted sequence `V^T S^(k+1) V`. It computes an
order basis with shift `(0^b,1^b)` using PM-Basis, and returns the sum of the
smallest `b` shifted row degrees. The stopping rule uses a gap of at least
10 between the two halves of the sorted degree list.

The mathematical reference is Pascal Skipness,
[*Modernizing Block Wiedemann: A Theoretical Framework Using Order Bases and
a Hybrid Rank Algorithm*](https://doi.org/10.3929/ethz-c-000800128), 2026:
Theorem 5.2 (p. 38) bounds the generator determinant degree of the shifted
sequence by `rank(S)`; Algorithm 5 and Theorem 5.6 (pp. 40–43) justify the
degree-based lower bound, including early stopping. The finite-field part of
that proof applies here directly. A favorable random projection is needed
for a tight bound, not for the lower-bound inequality. A bound equal to the
number of columns proves full column rank; an insufficient bound fails the
reproduction run instead of becoming a successful result.

These statements rely on exact finite-field arithmetic and correct sequence
and order-basis computation. Retained CUDA accumulation bounds and Rust NTT
coefficient/transform bounds must not be relaxed. Deformation arithmetic uses
`u32` residues, `u64` products, and primes at most 65521. The rank program
reads a completed WDM and fails if it lacks sufficient sequence data; it does
not poll a producer or reuse a historical sequence. Kernel recovery uses the
exported generator polynomials, then checks the recovered vectors directly.
The `_nullvectors_2.txt` output contains original source coordinates;
`_nullvectors_1.txt` contains preconditioned coordinates.

## Execution and source map

The reader interface is `./reproduce --genus G` or `./reproduce --all`, with
only `--gpus`, `--threads`, and `--output` as operational controls. Each genus
gets a fresh directory. GPU workers process independent base sectors; each
worker completes sequence generation, CPU rank analysis, and any kernel
recovery in that order. The CPU budget is shared between workers. Deformation
stages follow only after the required base jobs complete. Failure terminates
child processes and retains logs and working files; there is no resume,
download, or archived-sequence replay interface.

| Responsibility | Main implementation |
|---|---|
| Fixed cases, hashes, expected ranks | `data/cases.json` and `data/g*.json` |
| Execution, result checks, fresh summaries | `scripts/reproduce.py`, `Pipeline` |
| Independent section/product checks | `scripts/check_geometry.py` |
| Section spaces, tensors, sectors | `src/cyclic.rs`, `generate_cyclic_instance`, `verify_cyclic_instance` |
| Field, binary forms, linear solves | `src/field.rs`, `binary_form.rs`, `linear.rs`, `sections.rs` |
| Paired Taylor preparation | `scripts/prepare_deformation.py`, `build_instance` |
| Kernel selection, RHS, corrections, obstruction | `src/deformation.rs`, its four public stage functions |
| Base-sector sequence/recovery | `cuda/cuprym_cyclic.cu`, `cuprym_cyclic_kernel_vectors.cu` |
| Augmented/replacement sequence/recovery | `cuda/cuprym_deformation.cu`, `cuprym_deformation_kernel_vectors.cu` |
| Factored CUDA products and transposes | `cuda/cyclic_cuda_helpers.h`, `prym_cuda_helpers.h` |
| WDM interpretation and PM-Basis rank | `third_party/bcw/src/wdm_files.rs`, `rank_pipeline.rs`, `sigma_basis.rs` |
| Exact polynomial-matrix arithmetic | `third_party/bcw/src/poly_mat_mul.rs`, `ntt.rs`, `modular_linalg.rs` |

The developer tests include independent explicit g12 operators and transposes,
Taylor identities, and a complete Rust deformation-stage comparison against
that reference's kernel, corrections, and obstruction. Other tests check
corrupted inputs, incomplete stages, and known BCW ranks/file interfaces.
CPU tests establish those contracts; CUDA acceptance remains a separate
step on the author's server after pushing and cloning this repository.
