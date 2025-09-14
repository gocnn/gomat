package mat_test

import (
	"math/rand"
	"testing"

	"github.com/qntx/gomat/mat"
)

// DivNative is a pure Go implementation of element-wise division for float64 slices.
func DivNative(x1, x2, y []float64) {
	for i := range x1 {
		y[i] = x1[i] / x2[i]
	}
}

// TestDiv verifies the correctness of DivAVX, DivSSE, and DivNative.
func TestDiv(t *testing.T) {
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
			x1 := make([]float64, tc.n)
			x2 := make([]float64, tc.n)
			y := make([]float64, tc.n)
			yNative := make([]float64, tc.n)

			// Fill x1 and x2 with random values, ensuring x2 is not zero
			for i := range x1 {
				x1[i] = rand.Float64() * 100
				x2[i] = rand.Float64()*100 + 1
			}

			// Run all implementations
			mat.Div(x1, x2, y)
			DivNative(x1, x2, yNative)

			// Compare results
			for i := range x1 {
				if y[i] != yNative[i] {
					t.Errorf("Div failed: results differ from DivNative")
				}
			}
		})
	}
}

// BenchmarkDiv compares the performance of DivAVX, DivSSE, and DivNative for different slice sizes.
func BenchmarkDiv(b *testing.B) {
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
		x1 := make([]float64, size.n)
		x2 := make([]float64, size.n)
		y := make([]float64, size.n)

		// Fill x1 and x2 with random values, ensuring x2 is not zero
		for i := range x1 {
			x1[i] = rand.Float64() * 100
			x2[i] = rand.Float64()*100 + 1
		}

		// Benchmark Div
		b.Run("ASM/"+size.name, func(b *testing.B) {
			for b.Loop() {
				mat.Div(x1, x2, y)
			}
		})

		// Benchmark DivNative
		b.Run("Native/"+size.name, func(b *testing.B) {
			for b.Loop() {
				DivNative(x1, x2, y)
			}
		})
	}
}
