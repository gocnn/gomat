//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	add32 = AddSSE32
	add64 = AddSSE64
)

func init() {
	if hasAVX {
		add32 = AddAVX32
		add64 = AddAVX64
	}
}

// Add32 adds x1 and x2 element-wise, storing the result in y (32 bits).
func Add32(x1, x2, y []float32) {
	add32(x1, x2, y)
}

// Add64 adds x1 and x2 element-wise, storing the result in y (64 bits).
func Add64(x1, x2, y []float64) {
	add64(x1, x2, y)
}
