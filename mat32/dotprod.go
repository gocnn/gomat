//go:build amd64 && gc && !noasm && !gccgo

package mat32

var (
	dotProd = DotProdSSE
)

func init() {
	if hasAVX && hasFMA {
		dotProd = DotProdAVX
	}
}

// DotProd returns the dot product between x1 and x2 (32 bits).
func DotProd(x1, x2 []float32) float32 {
	return dotProd(x1, x2)
}
