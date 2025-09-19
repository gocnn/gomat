package cblas32

/*
#include "../cblas.h"
*/
import "C"
import (
	_ "unsafe"

	"github.com/gocnn/gomat/blas"
)

// Gemv computes
//
//	y = alpha * A * x + beta * y   if tA = blas.NoTrans
//	y = alpha * Aᵀ * x + beta * y  if tA = blas.Trans or blas.ConjTrans
//
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func Gemv(tA blas.Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cblas.h:200:6 void cblas_sgemv ...

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
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < max(1, n) {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(m-1)+n {
		panic(blas.ErrShortA)
	}
	var lenX, lenY int
	if tA == C.CblasNoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && len(x) <= (lenX-1)*incX) || (incX < 0 && len(x) <= (1-lenX)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (lenY-1)*incY) || (incY < 0 && len(y) <= (1-lenY)*incY) {
		panic(blas.ErrShortY)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_sgemv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_TRANSPOSE(tA), C.int(m), C.int(n), C.float(alpha), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX), C.float(beta), (*C.float)(_y), C.int(incY))
}

// Symv performs the matrix-vector operation
//
//	y = alpha * A * x + beta * y
//
// where A is an n×n symmetric matrix, x and y are vectors, and alpha and
// beta are scalars.
func Symv(ul blas.Uplo, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cblas.h:332:6 void cblas_ssymv ...

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
	if lda < max(1, n) {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+n {
		panic(blas.ErrShortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(blas.ErrShortY)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_ssymv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.int(n), C.float(alpha), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX), C.float(beta), (*C.float)(_y), C.int(incY))
}

// Trmv performs one of the matrix-vector operations
//
//	x = A * x   if tA == blas.NoTrans
//	x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix, and x is a vector.
func Trmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float32, lda int, x []float32, incX int) {
	// declared at cblas.h:210:6 void cblas_strmv ...

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
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < max(1, n) {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+n {
		panic(blas.ErrShortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	C.cblas_strmv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(n), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX))
}

// Trsv solves one of the systems of equations
//
//	A * x = b   if tA == blas.NoTrans
//	Aᵀ * x = b  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix, and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Trsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, a []float32, lda int, x []float32, incX int) {
	// declared at cblas.h:221:6 void cblas_strsv ...

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
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < max(1, n) {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+n {
		panic(blas.ErrShortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	C.cblas_strsv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(n), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX))
}

// Ger performs the rank-one operation
//
//	A += alpha * x * yᵀ
//
// where A is an m×n dense matrix, x and y are vectors, and alpha is a scalar.
func Ger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	// declared at cblas.h:344:6 void cblas_sger ...

	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if lda < max(1, n) {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (m-1)*incX) || (incX < 0 && len(x) <= (1-m)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(blas.ErrShortY)
	}
	if len(a) < lda*(m-1)+n {
		panic(blas.ErrShortA)
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	C.cblas_sger(C.CBLAS_LAYOUT(C.CblasRowMajor), C.int(m), C.int(n), C.float(alpha), (*C.float)(_x), C.int(incX), (*C.float)(_y), C.int(incY), (*C.float)(_a), C.int(lda))
}

// Syr performs the symmetric rank-one update
//
//	A += alpha * x * xᵀ
//
// where A is an n×n symmetric matrix, and x is a vector.
func Syr(ul blas.Uplo, n int, alpha float32, x []float32, incX int, a []float32, lda int) {
	// declared at cblas.h:347:6 void cblas_ssyr ...

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
	if lda < max(1, n) {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if len(a) < lda*(n-1)+n {
		panic(blas.ErrShortA)
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	C.cblas_ssyr(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.int(n), C.float(alpha), (*C.float)(_x), C.int(incX), (*C.float)(_a), C.int(lda))
}

// Syr2 performs the symmetric rank-two update
//
//	A += alpha * x * yᵀ + alpha * y * xᵀ
//
// where A is an n×n symmetric matrix, x and y are vectors, and alpha is a scalar.
func Syr2(ul blas.Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int) {
	// declared at cblas.h:353:6 void cblas_ssyr2 ...

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
	if lda < max(1, n) {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(blas.ErrShortY)
	}
	if len(a) < lda*(n-1)+n {
		panic(blas.ErrShortA)
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	C.cblas_ssyr2(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.int(n), C.float(alpha), (*C.float)(_x), C.int(incX), (*C.float)(_y), C.int(incY), (*C.float)(_a), C.int(lda))
}

// Gbmv performs one of the matrix-vector operations
//
//	y = alpha * A * x + beta * y   if tA == blas.NoTrans
//	y = alpha * Aᵀ * x + beta * y  if tA == blas.Trans or blas.ConjTrans
//
// where A is an m×n band matrix with kL sub-diagonals and kU super-diagonals,
// x and y are vectors, and alpha and beta are scalars.
func Gbmv(tA blas.Transpose, m, n, kL, kU int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cblas.h:205:6 void cblas_sgbmv ...

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
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if kL < 0 {
		panic(blas.ErrKLLT0)
	}
	if kU < 0 {
		panic(blas.ErrKULT0)
	}
	if lda < kL+kU+1 {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(min(m, n+kL)-1)+kL+kU+1 {
		panic(blas.ErrShortA)
	}
	var lenX, lenY int
	if tA == C.CblasNoTrans {
		lenX, lenY = n, m
	} else {
		lenX, lenY = m, n
	}
	if (incX > 0 && len(x) <= (lenX-1)*incX) || (incX < 0 && len(x) <= (1-lenX)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (lenY-1)*incY) || (incY < 0 && len(y) <= (1-lenY)*incY) {
		panic(blas.ErrShortY)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_sgbmv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_TRANSPOSE(tA), C.int(m), C.int(n), C.int(kL), C.int(kU), C.float(alpha), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX), C.float(beta), (*C.float)(_y), C.int(incY))
}

// Sbmv performs the matrix-vector operation
//
//	y = alpha * A * x + beta * y
//
// where A is an n×n symmetric band matrix with k super-diagonals, x and y are
// vectors, and alpha and beta are scalars.
func Sbmv(ul blas.Uplo, n, k int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cblas.h:336:6 void cblas_ssbmv ...

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
	if lda < k+1 {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+k+1 {
		panic(blas.ErrShortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(blas.ErrShortY)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_ssbmv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.int(n), C.int(k), C.float(alpha), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX), C.float(beta), (*C.float)(_y), C.int(incY))
}

// Tbmv performs one of the matrix-vector operations
//
//	x = A * x   if tA == blas.NoTrans
//	x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular band matrix with k+1 diagonals, and x is a vector.
func Tbmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float32, lda int, x []float32, incX int) {
	// declared at cblas.h:214:6 void cblas_stbmv ...

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
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < k+1 {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+k+1 {
		panic(blas.ErrShortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	C.cblas_stbmv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(n), C.int(k), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX))
}

// Tbsv solves one of the systems of equations
//
//	A * x = b   if tA == blas.NoTrans
//	Aᵀ * x = b  if tA == blas.Trans or tA == blas.ConjTrans
//
// where A is an n×n triangular band matrix with k+1 diagonals,
// and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Tbsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n, k int, a []float32, lda int, x []float32, incX int) {
	// declared at cblas.h:225:6 void cblas_stbsv ...

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
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	if lda < k+1 {
		panic(blas.ErrBadLdA)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(n-1)+k+1 {
		panic(blas.ErrShortA)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	var _a *float32
	if len(a) > 0 {
		_a = &a[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	C.cblas_stbsv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(n), C.int(k), (*C.float)(_a), C.int(lda), (*C.float)(_x), C.int(incX))
}

// Spmv performs the matrix-vector operation
//
//	y = alpha * A * x + beta * y
//
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha and beta are scalars.
func Spmv(ul blas.Uplo, n int, alpha float32, ap []float32, x []float32, incX int, beta float32, y []float32, incY int) {
	// declared at cblas.h:340:6 void cblas_sspmv ...

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
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(ap) < n*(n+1)/2 {
		panic(blas.ErrShortAP)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(blas.ErrShortY)
	}
	var _ap *float32
	if len(ap) > 0 {
		_ap = &ap[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_sspmv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.int(n), C.float(alpha), (*C.float)(_ap), (*C.float)(_x), C.int(incX), C.float(beta), (*C.float)(_y), C.int(incY))
}

// Tpmv performs one of the matrix-vector operations
//
//	x = A * x   if tA == blas.NoTrans
//	x = Aᵀ * x  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix in packed format, and x is a vector.
func Tpmv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []float32, x []float32, incX int) {
	// declared at cblas.h:218:6 void cblas_stpmv ...

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
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(ap) < n*(n+1)/2 {
		panic(blas.ErrShortAP)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	var _ap *float32
	if len(ap) > 0 {
		_ap = &ap[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	C.cblas_stpmv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(n), (*C.float)(_ap), (*C.float)(_x), C.int(incX))
}

// Tpsv solves one of the systems of equations
//
//	A * x = b   if tA == blas.NoTrans
//	Aᵀ * x = b  if tA == blas.Trans or blas.ConjTrans
//
// where A is an n×n triangular matrix in packed format, and x and b are vectors.
//
// At entry to the function, x contains the values of b, and the result is
// stored in-place into x.
//
// No test for singularity or near-singularity is included in this
// routine. Such tests must be performed before calling this routine.
func Tpsv(ul blas.Uplo, tA blas.Transpose, d blas.Diag, n int, ap []float32, x []float32, incX int) {
	// declared at cblas.h:229:6 void cblas_stpsv ...

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
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(ap) < n*(n+1)/2 {
		panic(blas.ErrShortAP)
	}
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	var _ap *float32
	if len(ap) > 0 {
		_ap = &ap[0]
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	C.cblas_stpsv(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.CBLAS_TRANSPOSE(tA), C.CBLAS_DIAG(d), C.int(n), (*C.float)(_ap), (*C.float)(_x), C.int(incX))
}

// Dspr performs the symmetric rank-one operation
//
//	A += alpha * x * xᵀ
//
// where A is an n×n symmetric matrix in packed format, x is a vector, and
// alpha is a scalar.
func Spr(ul blas.Uplo, n int, alpha float32, x []float32, incX int, ap []float32) {
	// declared at cblas.h:350:6 void cblas_sspr ...

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
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if len(ap) < n*(n+1)/2 {
		panic(blas.ErrShortAP)
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _ap *float32
	if len(ap) > 0 {
		_ap = &ap[0]
	}
	C.cblas_sspr(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.int(n), C.float(alpha), (*C.float)(_x), C.int(incX), (*C.float)(_ap))
}

// Spr2 performs the symmetric rank-2 update
//
//	A += alpha * x * yᵀ + alpha * y * xᵀ
//
// where A is an n×n symmetric matrix in packed format, x and y are vectors,
// and alpha is a scalar.
func Spr2(ul blas.Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, ap []float32) {
	// declared at cblas.h:357:6 void cblas_sspr2 ...

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
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(blas.ErrShortY)
	}
	if len(ap) < n*(n+1)/2 {
		panic(blas.ErrShortAP)
	}
	var _x *float32
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float32
	if len(y) > 0 {
		_y = &y[0]
	}
	var _a *float32
	if len(ap) > 0 {
		_a = &ap[0]
	}
	C.cblas_sspr2(C.CBLAS_LAYOUT(C.CblasRowMajor), C.CBLAS_UPLO(ul), C.int(n), C.float(alpha), (*C.float)(_x), C.int(incX), (*C.float)(_y), C.int(incY), (*C.float)(_a))
}
