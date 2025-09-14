//go:build !amd64 || !gc || noasm || gccgo

package mat

// Sum returns the sum of all values of x (64 bits).
func Sum(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	var y float64
	for _, v := range x {
		y += v
	}
	return y
}
