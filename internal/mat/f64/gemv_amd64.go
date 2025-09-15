//go:build !noasm && !gccgo && !safe

package f64

// GemvN computes
//
//	y = alpha * A * x + beta * y
//
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func GemvN(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr)

// GemvT computes
//
//	y = alpha * Aᵀ * x + beta * y
//
// where A is an m×n dense matrix, x and y are vectors, and alpha and beta are scalars.
func GemvT(m, n uintptr, alpha float64, a []float64, lda uintptr, x []float64, incX uintptr, beta float64, y []float64, incY uintptr)