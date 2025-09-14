//go:build amd64 && gc && !noasm && !gccgo

package mat32

var (
	sub = SubSSE
)

func init() {
	if hasAVX {
		sub = SubAVX
	}
}

// Sub subtracts x2 from x1, element-wise, storing the result in y (32 bits).
func Sub(x1, x2, y []float32) {
	sub(x1, x2, y)
}
