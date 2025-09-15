# BLAS (Basic Linear Algebra Subprograms)

This document describes the BLAS (Basic Linear Algebra Subprograms) specification and implementation. The specification is based on the reference implementation from Netlib, which is the original source of the BLAS standard and provides a widely-used and well-established implementation of the BLAS API.

See [https://www.netlib.org/blas/](https://www.netlib.org/blas/) for more information.

## BLAS Routines Documentation

### Level 1 BLAS: Vector, \( O(n) \) Operations

#### Vector Operations

| Type       | Name    | Arguments (Size)              | Description            | Equation          | FLOPs | Data  |
|------------|---------|--------------------------------|-----------------------|-------------------|-------|-------|
| s, d, c, z | axpy    | (n, alpha, x, incx, y, incy)  | Update vector          | \( y = y + \alpha x \) | \( 2n \) | \( 2n \) |
| s, d, c, z | scal    | (n, alpha, x, incx)           | Scale vector           | \( y = \alpha x \) | \( n \) | \( n \) |
| s, d, c, z | copy    | (n, x, incx, y, incy)         | Copy vector            | \( y = x \) | \( 0 \) | \( 2n \) |
| s, d, c, z | swap    | (n, x, incx, y, incy)         | Swap vectors           | \( x \leftrightarrow y \) | \( 0 \) | \( 2n \) |
| s, d       | dot     | (n, x, incx, y, incy)         | Dot product (real)     | \( x^T y \) | \( 2n \) | \( 2n \) |
| c, z       | dotu    | (n, x, incx, y, incy)         | Dot product (complex)  | \( x^T y \) | \( 2n \) | \( 2n \) |
| c, z       | dotc    | (n, x, incx, y, incy)         | Dot product (conjugate)| \( x^H y \) | \( 2n \) | \( 2n \) |
| sd, ds     | sdsdot  | (n, x, incx, y, incy)         | Dot product (double)   | \( x^T y \) | \( 2n \) | \( 2n \) |
| s, d, sc, dz | nrm2  | (n, x, incx)                  | 2-norm                 | \( \|x\|_2 \) | \( 2n \) | \( n \) |
| s, d, sc, dz | asum  | (n, x, incx)                  | 1-norm                 | \( \|Re(x)\|_1 + \|Im(x)\|_1 \) | \( n \) | \( n \) |
| s, d, c, z | iamax  | (n, x, incx)                  | \(\infty\)-norm        | \( \arg\max_j (Re(x_j) + Im(x_j)) \) | \( n \) | \( n \) |
| s, d, c, z | rotg   | (a, b, c, s)                  | Generate plane rotation (c real, s complex) | - | \( O(1) \) | \( O(1) \) |
| s, d, c, z† | rot   | (n, x, incx, y, incy, c, s)   | Apply plane rotation (c real, s complex) | - | \( 6n \) | \( 2n \) |
| c, z       | rotmg  | (d1, d2, a, b, param)         | Generate modified plane rotation | - | \( O(1) \) | \( O(1) \) |
| s, d       | rotm   | (n, x, incx, y, incy, param)  | Apply modified plane rotation | - | \( 6n \) | \( 2n \) |

- **Note**: † indicates routines added by LAPACK.

### Level 2 BLAS: Matrix-Vector, \( O(n^2) \) Operations

#### Full Storage

| Type       | Name    | Arguments (Size)              | Description            | Equation          | FLOPs | Data  |
|------------|---------|--------------------------------|-----------------------|-------------------|-------|-------|
| s, d, c, z | gemv    | (trans, m, n, alpha, A, ldA, x, incx, beta, y, incy) | General matrix-vector multiply | \( y = \alpha A x + \beta y \) | \( 2mn \) | \( m^2 \) |
| s, d†      | symv    | (uplo, n, alpha, A, ldA, x, incx, beta, y, incy) | Symmetric matrix-vector multiply | \( y = \alpha A x + \beta y \) | \( 2n^2 \) | \( n^2/2 \) |
| c, z       | hemv    | (uplo, n, alpha, A, ldA, x, incx, beta, y, incy) | Hermitian matrix-vector multiply | \( y = \alpha A x + \beta y \) | \( 2n^2 \) | \( n^2/2 \) |
| s, d, c, z | trmv    | (uplo, trans, diag, n, A, ldA, x, incx) | Triangular matrix-vector multiply | \( x = A x \) | \( n^2 \) | \( n^2/2 \) |
| s, d, c, z | trsv    | (uplo, trans, diag, n, A, ldA, x, incx) | Triangular matrix solve vector | \( x = A^{-1} x \) | \( n^2 \) | \( n^2/2 \) |
| s, d       | ger     | (m, n, alpha, x, incx, y, incy, A, ldA) | General rank-1 update | \( A = A + \alpha x y^T \) | \( 2mn \) | \( mn \) |
| c, z       | geru    | (m, n, alpha, x, incx, y, incy, A, ldA) | General rank-1 update (complex) | \( A = A + \alpha x y^T \) | \( 2mn \) | \( mn \) |
| c, z       | gerc    | (m, n, alpha, x, incx, y, incy, A, ldA) | General rank-1 update (conjugate) | \( A = A + \alpha x y^H \) | \( 2mn \) | \( mn \) |
| s, d†      | syr     | (uplo, n, alpha, x, incx, A, ldA) | Symmetric rank-1 update | \( A = A + \alpha x x^T \) | \( n^2 \) | \( n^2/2 \) |
| c, z       | her     | (uplo, n, alpha, x, incx, A, ldA) | Hermitian rank-1 update | \( A = A + \alpha x x^H \) | \( n^2 \) | \( n^2/2 \) |
| s, d, c, z | syr2    | (uplo, n, alpha, x, incx, y, incy, A, ldA) | Symmetric rank-2 update | \( A = A + \alpha x y^T + \alpha y x^T \) | \( 2n^2 \) | \( n^2/2 \) |
| c, z       | her2    | (uplo, n, alpha, x, incx, y, incy, A, ldA) | Hermitian rank-2 update | \( A = A + \alpha x y^H + \overline{\alpha} y x^H \) | \( 2n^2 \) | \( n^2/2 \) |

#### Band Storage

| Type       | Name    | Arguments (Size)              | Description            | Equation          | FLOPs | Data  |
|------------|---------|--------------------------------|-----------------------|-------------------|-------|-------|
| s, d, c, z | gbmv    | (trans, m, n, kl, ku, alpha, A, ldA, x, incx, beta, y, incy) | Band general matrix-vector multiply | \( y = \alpha A x + \beta y \) | \( 2mnk \) | \( mk + nk + mn \) |
| s, d†      | sbmv    | (uplo, n, k, alpha, A, ldA, x, incx, beta, y, incy) | Band symmetric matrix-vector multiply | \( y = \alpha A x + \beta y \) | \( 2n^2 \) | \( n^2/2 \) |
| s, d, c, z | tbmv    | (uplo, trans, diag, n, k, A, ldA, x, incx) | Band triangular matrix-vector multiply | \( x = A x \) | \( n^2 \) | \( n^2/2 \) |
| s, d, c, z | tbsv    | (uplo, trans, diag, n, k, A, ldA, x, incx) | Band triangular matrix solve vector | \( x = A^{-1} x \) | \( n^2 \) | \( n^2/2 \) |

#### Packed Storage

| Type       | Name    | Arguments (Size)              | Description            | Equation          | FLOPs | Data  |
|------------|---------|--------------------------------|-----------------------|-------------------|-------|-------|
| s, d, c, z | hpmv    | (uplo, n, alpha, Ap, x, incx, beta, y, incy) | Packed Hermitian matrix-vector multiply | \( y = \alpha A x + \beta y \) | \( 2n^2 \) | \( n^2/2 \) |
| s, d†      | spmv    | (uplo, n, alpha, Ap, x, incx, beta, y, incy) | Packed symmetric matrix-vector multiply | \( y = \alpha A x + \beta y \) | \( 2n^2 \) | \( n^2/2 \) |
| s, d, c, z | tpmv    | (uplo, trans, diag, n, Ap, x, incx) | Packed triangular matrix-vector multiply | \( x = A x \) | \( n^2 \) | \( n^2/2 \) |
| s, d, c, z | tpsv    | (uplo, trans, diag, n, Ap, x, incx) | Packed triangular matrix solve vector | \( x = A^{-1} x \) | \( n^2 \) | \( n^2/2 \) |
| s, d†      | spr     | (uplo, n, alpha, x, incx, Ap)  | Packed symmetric rank-1 update | \( A = A + \alpha x x^T \) | \( n^2 \) | \( n^2/2 \) |
| c, z       | hpr     | (uplo, n, alpha, x, incx, Ap)  | Packed Hermitian rank-1 update | \( A = A + \alpha x x^H \) | \( n^2 \) | \( n^2/2 \) |
| s, d, c, z | spr2    | (uplo, n, alpha, x, incx, y, incy, Ap) | Packed symmetric rank-2 update | \( A = A + \alpha x y^T + \alpha y x^T \) | \( 2n^2 \) | \( n^2/2 \) |
| c, z       | hpr2    | (uplo, n, alpha, x, incx, y, incy, Ap) | Packed Hermitian rank-2 update | \( A = A + \alpha x y^H + \overline{\alpha} y x^H \) | \( 2n^2 \) | \( n^2/2 \) |

### Level 3 BLAS: Matrix-Matrix, \( O(n^3) \) Operations

| Type       | Name    | Arguments (Size)              | Description            | Equation          | FLOPs | Data  |
|------------|---------|--------------------------------|-----------------------|-------------------|-------|-------|
| s, d, c, z | gemm    | (transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC) | General matrix-matrix multiply | \( C = \alpha A B + \beta C \) | \( 2mnk \) | \( mk + nk + mn \) |
| s, d, c, z | symm    | (uplo, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC) | Symmetric matrix-matrix multiply | \( C = \alpha A B + \beta C \) (or \( C = \alpha B A + \beta C \)) | \( 2mnk \) | \( m^2 + mn \) (left) |
| c, z       | hemm    | (uplo, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC) | Hermitian matrix-matrix multiply | \( C = \alpha A B + \beta C \) | \( 2mnk \) | \( m^2 + mn \) (left) |
| s, d, c, z | trmm    | (side, uplo, transA, diag, m, n, alpha, A, ldA, B, ldB) | Triangular matrix-matrix multiply | \( B = \alpha A B \) (or \( B = \alpha B A \)) | \( mn^2 \) | \( m^2 + mn \) (left) |
| s, d, c, z | trsm    | (side, uplo, transA, diag, m, n, alpha, A, ldA, B, ldB) | Triangular matrix solve matrix | \( B = \alpha A^{-1} B \) (or \( B = \alpha B A^{-1} \)) | \( mn^2 \) | \( m^2 + mn \) (left) |
| s, d, c, z | syrk    | (uplo, trans, n, k, alpha, A, ldA, beta, C, ldC) | Symmetric rank-k update | \( C = \alpha A A^T + \beta C \) | \( kn^2 \) | \( n^2/2 \) |
| s, d, c, z | herk    | (uplo, trans, n, k, alpha, A, ldA, beta, C, ldC) | Hermitian rank-k update | \( C = \alpha A A^H + \beta C \) | \( kn^2 \) | \( n^2/2 \) |
| s, d, c, z | syr2k   | (uplo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC) | Symmetric rank-2k update | \( C = \alpha A B^T + \alpha B A^T + \beta C \) | \( 2kn^2 \) | \( n^2/2 \) |
| c, z       | her2k   | (uplo, trans, n, k, alpha, A, ldA, B, ldB, beta, C, ldC) | Hermitian rank-2k update | \( C = \alpha A B^H + \overline{\alpha} B A^H + \beta C \) | \( 2kn^2 \) | \( n^2/2 \) |

- **Notes**:
  - Types: `s` (single precision real), `d` (double precision real), `c` (single precision complex), `z` (double precision complex), `sd/ds` (mixed precision).
  - † indicates routines added by LAPACK.
  - FLOPs and data sizes are approximate and depend on matrix dimensions and storage type.

## License

This BLAS implementation is based on the reference BLAS from `Netlib`, which is in the public domain.

### Netlib License

The original BLAS (Basic Linear Algebra Subprograms) from `Netlib` is available at:

- See: [http://www.netlib.org/blas/](http://www.netlib.org/blas/)
- LAPACK: [http://www.netlib.org/lapack/](http://www.netlib.org/lapack/)
- BLAS Technical Forum: [http://www.netlib.org/blas/blast-forum/](http://www.netlib.org/blas/blast-forum/)
- The reference BLAS is in the public domain and freely available for use.

### Gonum License

This Go implementation follows the Gonum project licensing:

- Copyright 2015 The Gonum Authors. All rights reserved.
- Use of this source code is governed by a BSD-style license that can be found in the LICENSE file.
- See: [https://github.com/gonum/gonum/blob/master/LICENSE](https://github.com/gonum/gonum/blob/master/LICENSE)
