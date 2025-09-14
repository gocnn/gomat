//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	dotProd32 = DotProdSSE32
	dotProd64 = DotProdSSE64
)

func init() {
	if hasAVX && hasFMA {
		dotProd32 = DotProdAVX32
		dotProd64 = DotProdAVX64
	}
}

// DotProd32 returns the dot product between x1 and x2 (32 bits).
func DotProd32(x1, x2 []float32) float32 {
	return dotProd32(x1, x2)
}

// DotProd64 returns the dot product between x1 and x2 (64 bits).
func DotProd64(x1, x2 []float64) float64 {
	return dotProd64(x1, x2)
}
