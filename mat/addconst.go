//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	addConst = AddConstSSE
)

func init() {
	if hasAVX {
		addConst = AddConstAVX
	}
}

// AddConst adds a constant value c to each element of x, storing the result in y (64 bits).
func AddConst(c float64, x, y []float64) {
	addConst(c, x, y)
}
