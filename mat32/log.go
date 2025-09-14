//go:build amd64 && gc && !noasm && !gccgo

package mat32

import "math"

// Log computes the natural logarithm of each element of x, storing the result in y (32 bits).
func Log(x, y []float32) {
	if len(x) == 0 {
		return
	}

	if hasAVX2 {
		_ = y[len(x)-1]
		max := len(x) - 8
		for i := 0; i <= max; i += 8 {
			LogAVX(x[i:], y[i:])
		}

		if max > 0 {
			for i, v := range x[:max+1] {
				if v == 0.0 {
					y[i] = float32(math.Inf(-1))
				}
			}
		}

		mod := len(x) % 8
		if mod > 0 {
			tailStart := len(x) - mod
			log(x[tailStart:], y[tailStart:])
		}
		return
	}

	_ = y[len(x)-1]
	max := len(x) - 4
	for i := 0; i <= max; i += 4 {
		LogSSE(x[i:], y[i:])
	}

	if max > 0 {
		for i, v := range x[:max+1] {
			if v == 0.0 {
				y[i] = float32(math.Inf(-1))
			}
		}
	}

	mod := len(x) % 4
	if mod > 0 {
		tailStart := len(x) - mod
		log(x[tailStart:], y[tailStart:])
	}
}

func log[F float32 | float64](x, y []F) {
	if len(x) == 0 {
		return
	}
	_ = y[len(x)-1]
	for i, xv := range x {
		y[i] = F(math.Log(float64(xv)))
	}
}
