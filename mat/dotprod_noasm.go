//go:build !amd64 || !gc || noasm || gccgo

package mat

// DotProd returns the dot product between x1 and x2 (64 bits).
func DotProd(x1, x2 []float64) float64 {
	if len(x1) == 0 {
		return 0
	}
	_ = x2[len(x1)-1]
	var y float64
	for i, x1v := range x1 {
		y += x1v * x2[i]
	}
	return y
}
