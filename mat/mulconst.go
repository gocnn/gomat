//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	mulConst = MulConstSSE
)

func init() {
	if hasAVX2 {
		mulConst = MulConstAVX
	}
}

// MulConst multiplies each element of x by a constant value c, storing the result in y (64 bits).
func MulConst(c float64, x, y []float64) {
	mulConst(c, x, y)
}
