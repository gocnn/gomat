//go:build cblas

package blas32

import (
	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/cblas/cblas32"
)

// Gemv computes
//
//	y = alpha * A * x + beta * y   if tA = blas.NoTrans
//	y = alpha * Aᵀ * x + beta * y  if tA = blas.Trans or blas.ConjTrans
//
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func Gemv(tA blas.Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	cblas32.Gemv(tA, m, n, alpha, a, lda, x, incX, beta, y, incY)
}

// Symv computes
//
//	y = alpha * A * x + beta * y
//
// where A is an n×n symmetric matrix, x and y are vectors, and alpha and
// beta are scalars.
func Symv(ul blas.Uplo, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	cblas32.Symv(ul, n, alpha, a, lda, x, incX, beta, y, incY)
}

// Trmv computes
//
//	x = A * x   if tA == blas.NoTrans
//	x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix, and x is a vector.
func Trmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float32, lda int, x []float32, incX int) {
	cblas32.Trmv(ul, tA, d, n, a, lda, x, incX)
}

// Trsv solves
//
//	A * x = b   if tA == blas.NoTrans
//	Aᵀ * x = b  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix and x is a vector, with the result stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Trsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float32, lda int, x []float32, incX int) {
	cblas32.Trsv(ul, tA, d, n, a, lda, x, incX)
}

// Ger performs a rank-1 update
//
//	A += alpha * x * yᵀ
//
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	cblas32.Ger(m, n, alpha, x, incX, y, incY, a, lda)
}

// Syr performs a rank-1 update
//
//	A += alpha * x * xᵀ
//
// where A is an n×n symmetric matrix, and x is a vector.
func Syr(ul blas.Uplo, n int, alpha float32, x []float32, incX int, a []float32, lda int) {
	cblas32.Syr(ul, n, alpha, x, incX, a, lda)
}

// Syr2 performs a rank-2 update
//
//	A += alpha * x * yᵀ + alpha * y * xᵀ
//
// where A is an n×n symmetric matrix, x and y are vectors, and alpha is a scalar.
func Syr2(ul blas.Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	cblas32.Syr2(ul, n, alpha, x, incX, y, incY, a, lda)
}

// Gbmv computes
//
//	y = alpha * A * x + beta * y   if tA == blas.NoTrans
//	y = alpha * Aᵀ * x + beta * y  if tA == blas.Trans or blas.ConjTrans
//
// where A is an m×n band matrix with kL sub-diagonals and kU super-diagonals,
// x and y are vectors, and alpha and beta are scalars.
func Gbmv(tA blas.Transpose, m, n, kL, kU int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	cblas32.Gbmv(tA, m, n, kL, kU, alpha, a, lda, x, incX, beta, y, incY)
}

// Sbmv computes
//
//	y = alpha * A * x + beta * y
//
// where A is an n×n symmetric band matrix with k super-diagonals, x and y are
// vectors, and alpha and beta are scalars.
func Sbmv(ul blas.Uplo, n, k int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	cblas32.Sbmv(ul, n, k, alpha, a, lda, x, incX, beta, y, incY)
}

// Tbmv computes
//
//	x = A * x   if tA == blas.NoTrans
//	x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular band matrix with k+1 diagonals, and x is a vector.
func Tbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float32, lda int, x []float32, incX int) {
	cblas32.Tbmv(ul, tA, d, n, k, a, lda, x, incX)
}

// Tbsv solves
//
//	A * x = b   if tA == blas.NoTrans
//	Aᵀ * x = b  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular band matrix with k+1 diagonals, and x is a vector,
// with the result stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Tbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float32, lda int, x []float32, incX int) {
	cblas32.Tbsv(ul, tA, d, n, k, a, lda, x, incX)
}

// Spmv computes
//
//	y = alpha * A * x + beta * y,
//
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha and beta are scalars.
func Spmv(ul blas.Uplo, n int, alpha float32, ap []float32, x []float32, incX int, beta float32, y []float32, incY int) {
	cblas32.Spmv(ul, n, alpha, ap, x, incX, beta, y, incY)
}

// Tpmv computes
//
//	x = A * x   if tA == blas.NoTrans
//	x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix in packed format, and x is a vector.
func Tpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []float32, x []float32, incX int) {
	cblas32.Tpmv(ul, tA, d, n, ap, x, incX)
}

// Tpsv solves
//
//	A * x = b   if tA == blas.NoTrans
//	Aᵀ * x = b  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix in packed format, and x is a vector,
// with the result stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Tpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []float32, x []float32, incX int) {
	cblas32.Tpsv(ul, tA, d, n, ap, x, incX)
}

// Spr performs a rank-1 update
//
//	A += alpha * x * xᵀ
//
// where A is an n×n symmetric matrix in packed format, x is a vector, and
// alpha is a scalar.
func Spr(ul blas.Uplo, n int, alpha float32, x []float32, incX int, ap []float32) {
	cblas32.Spr(ul, n, alpha, x, incX, ap)
}

// Spr2 performs a rank-2 update
//
//	A += alpha * x * yᵀ + alpha * y * xᵀ
//
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha is a scalar.
func Spr2(ul blas.Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, ap []float32) {
	cblas32.Spr2(ul, n, alpha, x, incX, y, incY, ap)
}
