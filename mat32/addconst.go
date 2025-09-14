//go:build amd64 && gc && !noasm && !gccgo

package mat32

var (
	addConst = AddConstSSE
)

func init() {
	if hasAVX {
		addConst = AddConstAVX
	}
}

// AddConst adds a constant value c to each element of x, storing the result in y (32 bits).
func AddConst(c float32, x, y []float32) {
	addConst(c, x, y)
}
