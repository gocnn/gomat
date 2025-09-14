//go:build !amd64 || !gc || noasm || gccgo

package mat

// Sub subtracts x2 from x1, element-wise, storing the result in y (64 bits).
func Sub(x1, x2, y []float64) {
	if len(x1) == 0 {
		return
	}
	_ = y[len(x1)-1]
	_ = x2[len(x1)-1]
	for i, x1v := range x1 {
		y[i] = x1v - x2[i]
	}
}
