//go:build !amd64 || !gc || noasm || gccgo

package mat

// Exp computes the base-e exponential of each element of x, storing the result in y (64 bits).
func Exp(x, y []float64) {
	exp(x, y)
}
