//go:build amd64 && gc && !noasm && !gccgo

package mat32

var (
	sum = SumSSE
)

func init() {
	if hasAVX {
		sum = SumAVX
	}
}

// Sum returns the sum of all values of x (32 bits).
func Sum(x []float32) float32 {
	return sum(x)
}
