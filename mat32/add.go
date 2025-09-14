//go:build amd64 && gc && !noasm && !gccgo

package mat32

var (
	add = AddSSE
)

func init() {
	if hasAVX {
		add = AddAVX
	}
}

// Add adds x1 and x2 element-wise, storing the result in y (32 bits).
func Add(x1, x2, y []float32) {
	add(x1, x2, y)
}
