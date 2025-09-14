//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	addConst32 = AddConstSSE32
	addConst64 = AddConstSSE64
)

func init() {
	if hasAVX2 {
		addConst32 = AddConstAVX32
		addConst64 = AddConstAVX64
	}
}

// AddConst32 adds a constant value c to each element of x, storing the result in y (32 bits).
func AddConst32(c float32, x, y []float32) {
	addConst32(c, x, y)
}

// AddConst64 adds a constant value c to each element of x, storing the result in y (64 bits).
func AddConst64(c float64, x, y []float64) {
	addConst64(c, x, y)
}
