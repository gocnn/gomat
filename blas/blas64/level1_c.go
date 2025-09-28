//go:build cblas

package blas64

import (
	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/cblas/cblas64"
)

// Axpy adds alpha times x to y
//
//	y[i] += alpha * x[i] for all i
func Axpy(n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	cblas64.Axpy(n, alpha, x, incX, y, incY)
}

// Scal scales x by alpha.
//
//	x[i] *= alpha
//
// Scal has no effect if incX < 0.
func Scal(n int, alpha float64, x []float64, incX int) {
	cblas64.Scal(n, alpha, x, incX)
}

// Copy copies the elements of x into the elements of y.
//
//	y[i] = x[i] for all i
func Copy(n int, x []float64, incX int, y []float64, incY int) {
	cblas64.Copy(n, x, incX, y, incY)
}

// Swap exchanges the elements of two vectors.
//
//	x[i], y[i] = y[i], x[i] for all i
func Swap(n int, x []float64, incX int, y []float64, incY int) {
	cblas64.Swap(n, x, incX, y, incY)
}

// Dot computes the dot product of the two vectors
//
//	\sum_i x[i]*y[i]
func Dot(n int, x []float64, incX int, y []float64, incY int) float64 {
	return cblas64.Dot(n, x, incX, y, incY)
}

// Nrm2 computes the Euclidean norm of a vector,
//
//	sqrt(\sum_i x[i] * x[i]).
//
// This function returns 0 if incX is negative.
func Nrm2(n int, x []float64, incX int) float64 {
	return cblas64.Nrm2(n, x, incX)
}

// Asum computes the sum of the absolute values of the elements of x.
//
//	\sum_i |x[i]|
//
// Asum returns 0 if incX is negative.
func Asum(n int, x []float64, incX int) float64 {
	return cblas64.Asum(n, x, incX)
}

// Iamax returns the index of an element of x with the largest absolute value.
// If there are multiple such indices the earliest is returned.
// Iamax returns -1 if n == 0.
func Iamax(n int, x []float64, incX int) int {
	return cblas64.Iamax(n, x, incX)
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
	return cblas64.Rotg(a, b)
}

// Rot applies a plane transformation.
//
//	x[i] = c * x[i] + s * y[i]
//	y[i] = c * y[i] - s * x[i]
func Rot(n int, x []float64, incX int, y []float64, incY int, c float64, s float64) {
	cblas64.Rot(n, x, incX, y, incY, c, s)
}

// Rotmg computes the modified Givens rotation. See
// http://www.netlib.org/lapack/explore-html/df/deb/drotmg_8f.html
// for more details.
func Rotmg(d1, d2, x1, y1 float64) (p blas.DrotmParams, rd1, rd2, rx1 float64) {
	return cblas64.Rotmg(d1, d2, x1, y1)
}

// Rotm applies the modified Givens rotation to the 2×n matrix.
func Rotm(n int, x []float64, incX int, y []float64, incY int, p blas.DrotmParams) {
	cblas64.Rotm(n, x, incX, y, incY, p)
}
