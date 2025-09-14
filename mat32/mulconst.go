//go:build amd64 && gc && !noasm && !gccgo

package mat32

var (
	mulConst = MulConstSSE
)

func init() {
	if hasAVX2 {
		mulConst = MulConstAVX
	}
}

// MulConst multiplies each element of x by a constant value c, storing the result in y (32 bits).
func MulConst(c float32, x, y []float32) {
	mulConst(c, x, y)
}
