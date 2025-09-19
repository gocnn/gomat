package main

import (
	"fmt"

	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/cblas/cblas32"
	"github.com/gocnn/gomat/cblas/cblas64"
)

func main() {
	a64 := []float64{1, 2, 3, 4}
	b64 := []float64{5, 6, 7, 8}
	c64 := []float64{0, 0, 0, 0}

	cblas64.Gemm(blas.NoTrans, blas.NoTrans, 2, 2, 2, 1.0, a64, 2, b64, 2, 0.0, c64, 2)

	fmt.Println(c64)

	a32 := []float32{1, 2, 3, 4}
	b32 := []float32{5, 6, 7, 8}
	c32 := []float32{0, 0, 0, 0}

	cblas32.Gemm(blas.NoTrans, blas.NoTrans, 2, 2, 2, 1.0, a32, 2, b32, 2, 0.0, c32, 2)

	fmt.Println(c32)
}
