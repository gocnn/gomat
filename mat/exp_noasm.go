//go:build !amd64 || !gc || noasm || gccgo

package mat

// Exp32 computes the base-e exponential of each element of x, storing the result in y (32 bits).
func Exp32(x, y []float32) {
	exp(x, y)
}

// Exp64 computes the base-e exponential of each element of x, storing the result in y (64 bits).
func Exp64(x, y []float64) {
	exp(x, y)
}
