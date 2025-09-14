//go:build amd64 && gc && !noasm && !gccgo

package mat

// Log computes the natural logarithm of each element of x, storing the result in y (64 bits).
func Log(x, y []float64) {
	log(x, y)
}
