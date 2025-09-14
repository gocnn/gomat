//go:build !amd64 || !gc || noasm || gccgo

package mat

import "math"

func log[F float32 | float64](x, y []F) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = F(math.Log(float64(xv)))
	}
}

// Log32 computes the natural logarithm of each element of x, storing the result in y (32 bits).
func Log32(x, y []float32) {
	log(x, y)
}

// Log64 computes the natural logarithm of each element of x, storing the result in y (64 bits).
func Log64(x, y []float64) {
	log(x, y)
}
