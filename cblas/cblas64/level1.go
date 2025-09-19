package cblas64

/*
#cgo CFLAGS: -g -O2
#include "../cblas.h"
*/
import "C"

import (
	"unsafe"

	"github.com/qntx/gomat/blas"
)

// y[i] += alpha * x[i] for all i
func Axpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	// declared at cblas.h:112:6 void cblas_daxpy ...

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
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float64
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_daxpy(C.int(n), C.double(alpha), (*C.double)(_x), C.int(incX), (*C.double)(_y), C.int(incY))
}

// Scal scales x by alpha.
//
//	x[i] *= alpha
//
// Scal has no effect if incX < 0.
func Scal(n int, alpha float64, x []float64, incX int) {
	// declared at cblas.h:152:6 void cblas_dscal ...

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 || incX < 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(x) <= (n-1)*incX {
		panic(blas.ErrShortX)
	}
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	C.cblas_dscal(C.int(n), C.double(alpha), (*C.double)(_x), C.int(incX))
}

// Copy copies the elements of x into the elements of y.
//
//	y[i] = x[i] for all i
func Copy(n int, x []float64, incX int, y []float64, incY int) {
	// declared at cblas.h:110:6 void cblas_dcopy ...

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
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float64
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_dcopy(C.int(n), (*C.double)(_x), C.int(incX), (*C.double)(_y), C.int(incY))
}

// Swap exchanges the elements of two vectors.
//
//	x[i], y[i] = y[i], x[i] for all i
func Swap(n int, x []float64, incX int, y []float64, incY int) {
	// declared at cblas.h:108:6 void cblas_dswap ...

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
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float64
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_dswap(C.int(n), (*C.double)(_x), C.int(incX), (*C.double)(_y), C.int(incY))
}

// Dot computes the dot product of the two vectors
//
//	\sum_i x[i]*y[i]
func Dot(n int, x []float64, incX int, y []float64, incY int) float64 {
	// declared at cblas.h:51:8 double cblas_ddot ...

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
		return 0
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if (incX > 0 && len(x) <= (n-1)*incX) || (incX < 0 && len(x) <= (1-n)*incX) {
		panic(blas.ErrShortX)
	}
	if (incY > 0 && len(y) <= (n-1)*incY) || (incY < 0 && len(y) <= (1-n)*incY) {
		panic(blas.ErrShortY)
	}
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float64
	if len(y) > 0 {
		_y = &y[0]
	}
	return float64(C.cblas_ddot(C.int(n), (*C.double)(_x), C.int(incX), (*C.double)(_y), C.int(incY)))
}

// Nrm2 computes the Euclidean norm of a vector,
//
//	sqrt(\sum_i x[i] * x[i]).
//
// This function returns 0 if incX is negative.
func Nrm2(n int, x []float64, incX int) float64 {
	// declared at cblas.h:74:8 double cblas_dnrm2 ...

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 || incX < 0 {
		return 0
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(x) <= (n-1)*incX {
		panic(blas.ErrShortX)
	}
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	return float64(C.cblas_dnrm2(C.int(n), (*C.double)(_x), C.int(incX)))
}

// Asum computes the sum of the absolute values of the elements of x.
//
//	\sum_i |x[i]|
//
// Asum returns 0 if incX is negative.
func Asum(n int, x []float64, incX int) float64 {
	// declared at cblas.h:75:8 double cblas_dasum ...

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 || incX < 0 {
		return 0
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(x) <= (n-1)*incX {
		panic(blas.ErrShortX)
	}
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	return float64(C.cblas_dasum(C.int(n), (*C.double)(_x), C.int(incX)))
}

// Iamax returns the index of an element of x with the largest absolute value.
// If there are multiple such indices the earliest is returned.
// Iamax returns -1 if n == 0.
func Iamax(n int, x []float64, incX int) int {
	// declared at cblas.h:88:13 unsigned long cblas_idamax ...

	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}

	// Quick return if possible.
	if n == 0 || incX < 0 {
		return -1
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(x) <= (n-1)*incX {
		panic(blas.ErrShortX)
	}
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	return int(C.cblas_idamax(C.int(n), (*C.double)(_x), C.int(incX)))
}

// Rotg computes a plane rotation
//
//	⎡  c s ⎤ ⎡ a ⎤ = ⎡ r ⎤
//	⎣ -s c ⎦ ⎣ b ⎦   ⎣ 0 ⎦
//
// satisfying c^2 + s^2 = 1.
//
// The computation uses the formulas
//
//	sigma = sgn(a)    if |a| >  |b|
//	      = sgn(b)    if |b| >= |a|
//	r = sigma*sqrt(a^2 + b^2)
//	c = 1; s = 0      if r = 0
//	c = a/r; s = b/r  if r != 0
//	c >= 0            if |a| > |b|
//
// The subroutine also computes
//
//	z = s    if |a| > |b|,
//	  = 1/c  if |b| >= |a| and c != 0
//	  = 1    if c = 0
//
// This allows c and s to be reconstructed from z as follows:
//
//	If z = 1, set c = 0, s = 1.
//	If |z| < 1, set c = sqrt(1 - z^2) and s = z.
//	If |z| > 1, set c = 1/z and s = sqrt(1 - c^2).
//
// NOTE: There is a discrepancy between the reference implementation and the
// BLAS technical manual regarding the sign for r when a or b are zero. Rotg
// agrees with the definition in the manual and other common BLAS
// implementations.
func Rotg(a, b float64) (c, s, r, z float64) {
	C.cblas_drotg((*C.double)(&a), (*C.double)(&b), (*C.double)(&c), (*C.double)(&s))
	return c, s, a, b
}

// Rot applies a plane transformation.
//
//	x[i] = c * x[i] + s * y[i]
//	y[i] = c * y[i] - s * x[i]
func Rot(n int, x []float64, incX int, y []float64, incY int, c float64, s float64) {
	// declared at cblas.h:142:6 void cblas_drot ...

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
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float64
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_drot(C.int(n), (*C.double)(_x), C.int(incX), (*C.double)(_y), C.int(incY), C.double(c), C.double(s))
}

// Rotmg computes the modified Givens rotation. See
// http://www.netlib.org/lapack/explore-html/df/deb/drotmg_8f.html
// for more details.
func Rotmg(d1, d2, x1, y1 float64) (p blas.DrotmParams, rd1, rd2, rx1 float64) {
	C.cblas_drotmg((*C.double)(&d1), (*C.double)(&d2), (*C.double)(&x1), C.double(y1), (*C.double)(unsafe.Pointer(&p)))
	return p, d1, d2, x1
}

// Rotm applies the modified Givens rotation to the 2×n matrix.
func Rotm(n int, x []float64, incX int, y []float64, incY int, p blas.DrotmParams) {
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if incX == 0 {
		panic(blas.ErrZeroIncX)
	}
	if incY == 0 {
		panic(blas.ErrZeroIncY)
	}
	if p.Flag < blas.Identity || p.Flag > blas.Diagonal {
		panic(blas.ErrBadFlag)
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
	var _x *float64
	if len(x) > 0 {
		_x = &x[0]
	}
	var _y *float64
	if len(y) > 0 {
		_y = &y[0]
	}
	C.cblas_drotm(C.int(n), (*C.double)(_x), C.int(incX), (*C.double)(_y), C.int(incY), (*C.double)(unsafe.Pointer(&p)))
}
