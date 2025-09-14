//go:build amd64 && gc && !noasm && !gccgo

package mat

var (
	sum = SumSSE
)

func init() {
	if hasAVX {
		sum = SumAVX
	}
}

// Sum returns the sum of all values of x (64 bits).
func Sum(x []float64) float64 {
	return sum(x)
}
