//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	div = DivSSE
)

func init() {
	if hasAVX {
		div = DivAVX
	}
}

// Div divides x1 by x2, element-wise, storing the result in y (64 bits).
func Div(x1, x2, y []float64) {
	div(x1, x2, y)
}
