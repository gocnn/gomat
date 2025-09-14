//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	dotProd = DotProdSSE
)

func init() {
	if hasAVX && hasFMA {
		dotProd = DotProdAVX
	}
}

// DotProd returns the dot product between x1 and x2 (64 bits).
func DotProd(x1, x2 []float64) float64 {
	return dotProd(x1, x2)
}
