//go:build !amd64 || !gc || noasm || gccgo

package mat

// MulConst multiplies each element of x by a constant value c, storing the result in y (64 bits).
func MulConst(c float64, x, y []float64) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = xv * c
	}
}
