package mat

import "math"

func exp(x, y []float64) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = math.Exp(xv)
	}
}
