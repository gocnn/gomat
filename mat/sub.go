//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	sub = SubSSE
)

func init() {
	if hasAVX {
		sub = SubAVX
	}
}

// Sub subtracts x2 from x1, element-wise, storing the result in y (64 bits).
func Sub(x1, x2, y []float64) {
	sub(x1, x2, y)
}
