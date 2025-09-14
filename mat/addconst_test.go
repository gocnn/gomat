package mat_test

import (
	"math/rand"
	"testing"

	"github.com/qntx/gomat/mat"
)

// AddConstNative is a pure Go implementation of adding a constant to each element of x, storing in y.
func AddConstNative(c float64, x, y []float64) {
	for i := range x {
		y[i] = c + x[i]
	}
}

// TestAddConst verifies the correctness of AddConst.
func TestAddConst(t *testing.T) {
	// Test cases with different slice lengths
	tests := []struct {
		name string
		n    int
	}{
		{"Small", 8},
		{"Medium", 128},
		{"Large", 1024},
		{"NonAligned", 123}, // Non-aligned length to test edge cases
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Initialize input slices with random data
			c := rand.Float64() * 100
			x := make([]float64, tc.n)
			y := make([]float64, tc.n)
			yNative := make([]float64, tc.n)

			// Fill x with random values
			for i := range x {
				x[i] = rand.Float64() * 100
			}

			// Run all implementations
			mat.AddConst(c, x, y)
			AddConstNative(c, x, yNative)

			// Compare results
			for i := range x {
				if y[i] != yNative[i] {
					t.Errorf("AddConst failed: results differ from AddConstNative")
				}
			}
		})
	}
}

// BenchmarkAddConst compares the performance of AddConst implementations for different slice sizes.
func BenchmarkAddConst(b *testing.B) {
	// Different slice sizes to test
	sizes := []struct {
		name string
		n    int
	}{
		{"8", 8},         // Small
		{"128", 128},     // Medium
		{"1024", 1024},   // Large
		{"65536", 65536}, // Very large
		{"123", 123},     // Non-aligned
	}

	for _, size := range sizes {
		// Initialize input slices
		c := rand.Float64() * 100
		x := make([]float64, size.n)
		y := make([]float64, size.n)

		// Fill x with random values
		for i := range x {
			x[i] = rand.Float64() * 100
		}

		// Benchmark AddConst
		b.Run("ASM/"+size.name, func(b *testing.B) {
			for b.Loop() {
				mat.AddConst(c, x, y)
			}
		})

		// Benchmark AddConstNative
		b.Run("Native/"+size.name, func(b *testing.B) {
			for b.Loop() {
				AddConstNative(c, x, y)
			}
		})
	}
}
