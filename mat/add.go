//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	add = AddSSE
)

func init() {
	if hasAVX {
		add = AddAVX
	}
}

// Add adds x1 and x2 element-wise, storing the result in y (64 bits).
func Add(x1, x2, y []float64) {
	add(x1, x2, y)
}
