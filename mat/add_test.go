package mat_test

import (
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/qntx/gomat/mat"
)

// AddNative is a pure Go implementation of element-wise addition for float64 slices.
func AddNative(x1, x2, y []float64) {
	for i := range x1 {
		y[i] = x1[i] + x2[i]
	}
}

// TestAdd verifies the correctness of AddAVX, AddSSE, and AddNative.
func TestAdd(t *testing.T) {
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

			// Fill x1 and x2 with random values
			for i := range x1 {
				x1[i] = rand.Float64() * 100
				x2[i] = rand.Float64() * 100
			}

			// Run all implementations
			mat.Add(x1, x2, y)
			AddNative(x1, x2, yNative)

			// Compare results
			if !slices.Equal(y, yNative) {
				t.Errorf("Add failed: results differ from AddNative")
			}
		})
	}
}

// BenchmarkAdd compares the performance of AddAVX, AddSSE, and AddNative for different slice sizes.
func BenchmarkAdd(b *testing.B) {
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

		// Fill x1 and x2 with random values
		for i := range x1 {
			x1[i] = rand.Float64() * 100
			x2[i] = rand.Float64() * 100
		}

		// Benchmark AddAVX
		b.Run("ASM/"+size.name, func(b *testing.B) {
			for b.Loop() {
				mat.Add(x1, x2, y)
			}
		})

		// Benchmark AddNative
		b.Run("Native/ "+size.name, func(b *testing.B) {
			for b.Loop() {
				AddNative(x1, x2, y)
			}
		})
	}
}
