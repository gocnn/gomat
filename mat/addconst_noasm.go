//go:build !amd64 || !gc || noasm || gccgo

package mat

// AddConst adds a constant value c to each element of x, storing the result in y (64 bits).
func AddConst(c float64, x, y []float64) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = xv + c
	}
}
