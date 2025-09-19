// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lapack64

import (
	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/blas/blas64"
	"github.com/gocnn/gomat/lapack"
)

// Trtrs solves a triangular system of the form A * X = B or Aᵀ * X = B. Trtrs
// returns whether the solve completed successfully. If A is singular, no solve is performed.
func Trtrs(uplo blas.Uplo, trans blas.Transpose, diag blas.Diag, n, nrhs int, a []float64, lda int, b []float64, ldb int) (ok bool) {
	switch {
	case uplo != blas.Upper && uplo != blas.Lower:
		panic(lapack.ErrBadUplo)
	case trans != blas.NoTrans && trans != blas.Trans && trans != blas.ConjTrans:
		panic(lapack.ErrBadTrans)
	case diag != blas.NonUnit && diag != blas.Unit:
		panic(lapack.ErrBadDiag)
	case n < 0:
		panic(lapack.ErrNLT0)
	case nrhs < 0:
		panic(lapack.ErrNrhsLT0)
	case lda < max(1, n):
		panic(lapack.ErrBadLdA)
	case ldb < max(1, nrhs):
		panic(lapack.ErrBadLdB)
	}

	if n == 0 {
		return true
	}

	switch {
	case len(a) < (n-1)*lda+n:
		panic(lapack.ErrShortA)
	case len(b) < (n-1)*ldb+nrhs:
		panic(lapack.ErrShortB)
	}

	// Check for singularity.
	nounit := diag == blas.NonUnit
	if nounit {
		for i := 0; i < n; i++ {
			if a[i*lda+i] == 0 {
				return false
			}
		}
	}
	blas64.Trsm(blas.Left, uplo, trans, diag, n, nrhs, 1, a, lda, b, ldb)
	return true
}
