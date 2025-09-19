package cblas64

/*
#cgo CFLAGS: -g -O2
#include "../cblas.h"
*/
import "C"
import (
	_ "unsafe"

	"github.com/qntx/gomat/blas"
)

// Gemm performs one of the matrix-matrix operations
//
//	C = alpha * A * B + beta * C
//	C = alpha * Aᵀ * B + beta * C
//	C = alpha * A * Bᵀ + beta * C
//	C = alpha * Aᵀ * Bᵀ + beta * C
//
// where A is an m×k or k×m dense matrix, B is an n×k or k×n dense matrix, C is
// an m×n matrix, and alpha and beta are scalars. tA and tB specify whether A or
// B are transposed.
func Gemm(tA, tB blas.Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cblas.h:466:6 void cblas_dgemm ...

	switch tA {
	case blas.NoTrans:
		tA = C.CblasNoTrans
	case blas.Trans:
		tA = C.CblasTrans
	case blas.ConjTrans:
		tA = C.CblasConjTrans
	default:
		panic(blas.ErrBadTranspose)
	}
	switch tB {
	case blas.NoTrans:
		tB = C.CblasNoTrans
	case blas.Trans:
		tB = C.CblasTrans
	case blas.ConjTrans:
		tB = C.CblasConjTrans
	default:
		panic(blas.ErrBadTranspose)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	var rowA, colA, rowB, colB int
	if tA == C.CblasNoTrans {
		rowA, colA = m, k
	} else {
		rowA, colA = k, m
	}
	if tB == C.CblasNoTrans {
		rowB, colB = k, n
	} else {
		rowB, colB = n, k
	}
	if lda < max(1, colA) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, colB) {
		panic(blas.ErrBadLdB)
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(rowA-1)+colA {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(rowB-1)+colB {
		panic(blas.ErrShortB)
	}
	if len(c) < ldc*(m-1)+n {
		panic(blas.ErrShortC)
	}
	var _a *float64
	if len(a) > 0 {
		_a = &a[0]
	}
	var _b *float64
	if len(b) > 0 {
		_b = &b[0]
	}
	var _c *float64
	if len(c) > 0 {
		_c = &c[0]
	}
	C.cblas_dgemm(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_TRANSPOSE(tA), C.CBLAS_TRANSPOSE(tB), C.int(m), C.int(n), C.int(k), C.double(alpha), (*C.double)(_a), C.int(lda), (*C.double)(_b), C.int(ldb), C.double(beta), (*C.double)(_c), C.int(ldc))
}

// Symm performs one of the matrix-matrix operations
//
//	C = alpha * A * B + beta * C  if side == blas.Left
//	C = alpha * B * A + beta * C  if side == blas.Right
//
// where A is an n×n or m×m symmetric matrix, B and C are m×n matrices, and alpha
// is a scalar.
func Symm(s blas.Side, ul blas.Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cblas.h:471:6 void cblas_dsymm ...

	switch ul {
	case blas.Upper:
		ul = C.CblasUpper
	case blas.Lower:
		ul = C.CblasLower
	default:
		panic(blas.ErrBadUplo)
	}
	switch s {
	case blas.Left:
		s = C.CblasLeft
	case blas.Right:
		s = C.CblasRight
	default:
		panic(blas.ErrBadSide)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	var k int
	if s == C.CblasLeft {
		k = m
	} else {
		k = n
	}
	if lda < max(1, k) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, n) {
		panic(blas.ErrBadLdB)
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(blas.ErrShortB)
	}
	if len(c) < ldc*(m-1)+n {
		panic(blas.ErrShortC)
	}
	var _a *float64
	if len(a) > 0 {
		_a = &a[0]
	}
	var _b *float64
	if len(b) > 0 {
		_b = &b[0]
	}
	var _c *float64
	if len(c) > 0 {
		_c = &c[0]
	}
	C.cblas_dsymm(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_SIDE(s), C.CBLAS_UPLO(ul), C.int(m), C.int(n), C.double(alpha), (*C.double)(_a), C.int(lda), (*C.double)(_b), C.int(ldb), C.double(beta), (*C.double)(_c), C.int(ldc))
}

// Trmm performs one of the matrix-matrix operations
//
//	B = alpha * A * B   if tA == blas.NoTrans and side == blas.Left
//	B = alpha * Aᵀ * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//	B = alpha * B * A   if tA == blas.NoTrans and side == blas.Right
//	B = alpha * B * Aᵀ  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
//
// where A is an n×n or m×m triangular matrix, B is an m×n matrix, and alpha is a scalar.
func Trmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	// declared at cblas.h:485:6 void cblas_dtrmm ...

	switch tA {
	case blas.NoTrans:
		tA = C.CblasNoTrans
	case blas.Trans:
		tA = C.CblasTrans
	case blas.ConjTrans:
		tA = C.CblasConjTrans
	default:
		panic(blas.ErrBadTranspose)
	}
	switch ul {
	case blas.Upper:
		ul = C.CblasUpper
	case blas.Lower:
		ul = C.CblasLower
	default:
		panic(blas.ErrBadUplo)
	}
	switch d {
	case blas.NonUnit:
		d = C.CblasNonUnit
	case blas.Unit:
		d = C.CblasUnit
	default:
		panic(blas.ErrBadDiag)
	}
	switch s {
	case blas.Left:
		s = C.CblasLeft
	case blas.Right:
		s = C.CblasRight
	default:
		panic(blas.ErrBadSide)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	var k int
	if s == C.CblasLeft {
		k = m
	} else {
		k = n
	}
	if lda < max(1, k) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, n) {
		panic(blas.ErrBadLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(blas.ErrShortB)
	}
	var _a *float64
	if len(a) > 0 {
		_a = &a[0]
	}
	var _b *float64
	if len(b) > 0 {
		_b = &b[0]
	}
	C.cblas_dtrmm(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_SIDE(s), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(m), C.int(n), C.double(alpha), (*C.double)(_a), C.int(lda), (*C.double)(_b), C.int(ldb))
}

// Trsm solves one of the matrix equations
//
//	A * X = alpha * B   if tA == blas.NoTrans and side == blas.Left
//	Aᵀ * X = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//	X * A = alpha * B   if tA == blas.NoTrans and side == blas.Right
//	X * Aᵀ = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
//
// where A is an n×n or m×m triangular matrix, X and B are m×n matrices, and alpha is a
// scalar.
//
// At entry to the function, X contains the values of B, and the result is
// stored in-place into X.
//
// No check is made that A is invertible.
func Trsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int) {
	// declared at cblas.h:490:6 void cblas_dtrsm ...

	switch tA {
	case blas.NoTrans:
		tA = C.CblasNoTrans
	case blas.Trans:
		tA = C.CblasTrans
	case blas.ConjTrans:
		tA = C.CblasConjTrans
	default:
		panic(blas.ErrBadTranspose)
	}
	switch ul {
	case blas.Upper:
		ul = C.CblasUpper
	case blas.Lower:
		ul = C.CblasLower
	default:
		panic(blas.ErrBadUplo)
	}
	switch d {
	case blas.NonUnit:
		d = C.CblasNonUnit
	case blas.Unit:
		d = C.CblasUnit
	default:
		panic(blas.ErrBadDiag)
	}
	switch s {
	case blas.Left:
		s = C.CblasLeft
	case blas.Right:
		s = C.CblasRight
	default:
		panic(blas.ErrBadSide)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	var k int
	if s == C.CblasLeft {
		k = m
	} else {
		k = n
	}
	if lda < max(1, k) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, n) {
		panic(blas.ErrBadLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(blas.ErrShortB)
	}
	var _a *float64
	if len(a) > 0 {
		_a = &a[0]
	}
	var _b *float64
	if len(b) > 0 {
		_b = &b[0]
	}
	C.cblas_dtrsm(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_SIDE(s), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(m), C.int(n), C.double(alpha), (*C.double)(_a), C.int(lda), (*C.double)(_b), C.int(ldb))
}

// Syrk performs one of the symmetric rank-k operations
//
//	C = alpha * A * Aᵀ + beta * C  if tA == blas.NoTrans
//	C = alpha * Aᵀ * A + beta * C  if tA == blas.Trans or tA == blas.ConjTrans
//
// where A is an n×k or k×n matrix, C is an n×n symmetric matrix, and alpha and
// beta are scalars.
func Syrk(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int) {
	// declared at cblas.h:476:6 void cblas_dsyrk ...

	switch t {
	case blas.NoTrans:
		t = C.CblasNoTrans
	case blas.Trans:
		t = C.CblasTrans
	case blas.ConjTrans:
		t = C.CblasConjTrans
	default:
		panic(blas.ErrBadTranspose)
	}
	switch ul {
	case blas.Upper:
		ul = C.CblasUpper
	case blas.Lower:
		ul = C.CblasLower
	default:
		panic(blas.ErrBadUplo)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	var row, col int
	if t == C.CblasNoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda < max(1, col) {
		panic(blas.ErrBadLdA)
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(row-1)+col {
		panic(blas.ErrShortA)
	}
	if len(c) < ldc*(n-1)+n {
		panic(blas.ErrShortC)
	}
	var _a *float64
	if len(a) > 0 {
		_a = &a[0]
	}
	var _c *float64
	if len(c) > 0 {
		_c = &c[0]
	}
	C.cblas_dsyrk(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(t), C.int(n), C.int(k), C.double(alpha), (*C.double)(_a), C.int(lda), C.double(beta), (*C.double)(_c), C.int(ldc))
}

// Syr2k performs one of the symmetric rank 2k operations
//
//	C = alpha * A * Bᵀ + alpha * B * Aᵀ + beta * C  if tA == blas.NoTrans
//	C = alpha * Aᵀ * B + alpha * Bᵀ * A + beta * C  if tA == blas.Trans or tA == blas.ConjTrans
//
// where A and B are n×k or k×n matrices, C is an n×n symmetric matrix, and
// alpha and beta are scalars.
func Syr2k(ul blas.Uplo, t blas.Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int) {
	// declared at cblas.h:480:6 void cblas_dsyr2k ...

	switch t {
	case blas.NoTrans:
		t = C.CblasNoTrans
	case blas.Trans:
		t = C.CblasTrans
	case blas.ConjTrans:
		t = C.CblasConjTrans
	default:
		panic(blas.ErrBadTranspose)
	}
	switch ul {
	case blas.Upper:
		ul = C.CblasUpper
	case blas.Lower:
		ul = C.CblasLower
	default:
		panic(blas.ErrBadUplo)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	var row, col int
	if t == C.CblasNoTrans {
		row, col = n, k
	} else {
		row, col = k, n
	}
	if lda < max(1, col) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, col) {
		panic(blas.ErrBadLdB)
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(row-1)+col {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(row-1)+col {
		panic(blas.ErrShortB)
	}
	if len(c) < ldc*(n-1)+n {
		panic(blas.ErrShortC)
	}
	var _a *float64
	if len(a) > 0 {
		_a = &a[0]
	}
	var _b *float64
	if len(b) > 0 {
		_b = &b[0]
	}
	var _c *float64
	if len(c) > 0 {
		_c = &c[0]
	}
	C.cblas_dsyr2k(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(t), C.int(n), C.int(k), C.double(alpha), (*C.double)(_a), C.int(lda), (*C.double)(_b), C.int(ldb), C.double(beta), (*C.double)(_c), C.int(ldc))
}
