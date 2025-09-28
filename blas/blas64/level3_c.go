//go:build cblas

package blas64

import (
	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/cblas/cblas64"
)

// Gemm computes
//
//	C = alpha * A * B + beta * C
//
// where A is an m×k or k×m dense matrix, B is an n×k or k×n dense matrix, C is
// an m×n matrix, and alpha and beta are scalars. tA and tB specify whether A or
// B are transposed.
func Gemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	cblas64.Gemm(tA, tB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Symm computes
//
//	C = alpha * A * B + beta * C  if side == blas.Left
//	C = alpha * B * A + beta * C  if side == blas.Right
//
// where A is an n×n or m×m symmetric matrix, B and C are m×n matrices, and alpha
// is a scalar.
func Symm(s blas.Side, ul blas.Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	cblas64.Symm(s, ul, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
}

// Trmm computes
//
//	B = alpha * A * B   if tA == blas.NoTrans and side == blas.Left
//	B = alpha * Aᵀ * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//	B = alpha * B * A   if tA == blas.NoTrans and side == blas.Right
//	B = alpha * B * Aᵀ  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
//
// where A is an n×n or m×m triangular matrix, B is an m×n matrix, and alpha is a scalar.
func Trmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	cblas64.Trmm(s, ul, tA, d, m, n, alpha, a, lda, b, ldb)
}

// Trsm solves
//
//	A * X = alpha * B   if tA == blas.NoTrans and side == blas.Left
//	Aᵀ * X = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//	X * A = alpha * B   if tA == blas.NoTrans and side == blas.Right
//	X * Aᵀ = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
//
// where A is an n×n or m×m triangular matrix, X is an m×n matrix, and alpha is a scalar.
// The result X is stored in-place into B.
//
// No check is made that A is invertible.
func Trsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	cblas64.Trsm(s, ul, tA, d, m, n, alpha, a, lda, b, ldb)
}

// Syrk computes
//
//	C = alpha * A * Aᵀ + beta * C  if t == blas.NoTrans
//	C = alpha * Aᵀ * A + beta * C  if t == blas.Trans or blas.ConjTrans
//
// where A is an n×k or k×n matrix, C is an n×n symmetric matrix, and alpha and
// beta are scalars.
func Syrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	cblas64.Syrk(ul, t, n, k, alpha, a, lda, beta, c, ldc)
}

// Syr2k computes
//
//	C = alpha * A * Bᵀ + alpha * B * Aᵀ + beta * C  if t == blas.NoTrans
//	C = alpha * Aᵀ * B + alpha * Bᵀ * A + beta * C  if t == blas.Trans or blas.ConjTrans
//
// where A and B are n×k or k×n matrices, C is an n×n symmetric matrix, and
// alpha and beta are scalars.
func Syr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	cblas64.Syr2k(ul, t, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
}
