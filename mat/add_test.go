package mat_test

import (
	"testing"

	"github.com/qntx/gomat/mat"
)

func TestAdd(t *testing.T) {
	t.Parallel()

	x1 := make([]float64, 0, 2_000)
	x2 := make([]float64, 0, 2_000)
	expected := make([]float64, 0, 2_000)
	actual := make([]float64, 0, 2_000)

	for size := 0; size < 2_000; size++ {
		x1 = x1[:size]
		x2 = x2[:size]
		expected = expected[:size]
		actual = actual[:size]
		RandVec(x1)
		RandVec(x2)
		testingAdd(x1, x2, expected)

		mat.Add(x1, x2, actual)

		RequireSlicesInDelta(t, expected, actual, 1e-6)
	}

	// Test different alignments
	x1 = x1[:16]
	x2 = x2[:16]
	expected = expected[:16]
	actual = actual[:16]
	for offset := range x1 {
		testingAdd(x1[offset:], x2[offset:], expected[offset:])
		mat.Add(x1[offset:], x2[offset:], actual[offset:])
		RequireSlicesInDelta(t, expected[offset:], actual[offset:], 1e-6)
	}
}

func BenchmarkAdd(b *testing.B) {
	size := 1_000_000
	x1 := NewRandVec[float64](size)
	x2 := NewRandVec[float64](size)
	y := make([]float64, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mat.Add(x1, x2, y)
	}
}

func BenchmarkAddPureGo(b *testing.B) {
	size := 1_000_000
	x1 := NewRandVec[float64](size)
	x2 := NewRandVec[float64](size)
	y := make([]float64, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pureGoAdd(x1, x2, y)
	}
}

func pureGoAdd(x1, x2, y []float64) {
	for i, v1 := range x1 {
		y[i] = v1 + x2[i]
	}
}

func testingAdd(x1, x2, y []float64) {
	if len(x1) != len(x2) || len(x1) != len(y) {
		panic("len mismatch")
	}
	for i, x1v := range x1 {
		y[i] = x1v + x2[i]
	}
}
