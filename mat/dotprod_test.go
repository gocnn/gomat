package mat_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/qntx/gomat/mat"
)

// DotProdNative is a pure Go implementation of dot product.
func DotProdNative(x1, x2 []float64) float64 {
	sum := 0.0
	for i := range x1 {
		sum += x1[i] * x2[i]
	}
	return sum
}

// TestDotProd verifies the correctness of DotProd.
func TestDotProd(t *testing.T) {
	tests := []struct {
		name string
		n    int
	}{
		{"Small", 8},
		{"Medium", 128},
		{"Large", 1024},
		{"NonAligned", 123},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			x1 := make([]float64, tc.n)
			x2 := make([]float64, tc.n)
			for i := range x1 {
				x1[i] = rand.Float64() * 100
				x2[i] = rand.Float64() * 100
			}

			result := mat.DotProd(x1, x2)
			native := DotProdNative(x1, x2)

			tol := 1e-12
			if math.Abs(result-native) > tol*math.Max(math.Abs(result), math.Abs(native)) {
				t.Errorf("DotProd failed: result %v != native %v (diff %v)", result, native, result-native)
			}
		})
	}
}

// BenchmarkDotProd compares the performance of DotProd implementations for different slice sizes.
func BenchmarkDotProd(b *testing.B) {
	sizes := []struct {
		name string
		n    int
	}{
		{"8", 8},
		{"128", 128},
		{"1024", 1024},
		{"65536", 65536},
		{"123", 123},
	}

	for _, size := range sizes {
		x1 := make([]float64, size.n)
		x2 := make([]float64, size.n)
		for i := range x1 {
			x1[i] = rand.Float64() * 100
			x2[i] = rand.Float64() * 100
		}

		b.Run("ASM/"+size.name, func(b *testing.B) {
			for b.Loop() {
				mat.DotProd(x1, x2)
			}
		})

		b.Run("Native/"+size.name, func(b *testing.B) {
			for b.Loop() {
				DotProdNative(x1, x2)
			}
		})
	}
}
