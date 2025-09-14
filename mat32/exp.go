//go:build amd64 && gc && !noasm && !gccgo

package mat32

import "math"

// Exp32 computes the base-e exponential of each element of x, storing the result in y (32 bits).
func Exp32(x, y []float32) {
	if len(x) == 0 {
		return
	}

	if hasAVX2 {
		_ = y[len(x)-1]
		max := len(x) - 8
		for i := 0; i <= max; i += 8 {
			ExpAVX(x[i:], y[i:])
		}

		mod := len(x) % 8
		if mod > 0 {
			tailStart := len(x) - mod
			exp(x[tailStart:], y[tailStart:])
		}
		return
	}

	_ = y[len(x)-1]
	max := len(x) - 4
	for i := 0; i <= max; i += 4 {
		ExpSSE(x[i:], y[i:])
	}

	mod := len(x) % 4
	if mod > 0 {
		tailStart := len(x) - mod
		exp(x[tailStart:], y[tailStart:])
	}
}

func exp[F float32 | float64](x, y []F) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = F(math.Exp(float64(xv)))
	}
}
