//go:build !cblas

// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blas32

import (
	"runtime"
	"sync"

	"github.com/gocnn/gomat/blas"
	"github.com/gocnn/gomat/internal/mat/f32"
)

// Gemm performs one of the matrix-matrix operations
//
//	C = alpha * A * B + beta * C
//	C = alpha * Aᵀ * B + beta * C
//	C = alpha * A * Bᵀ + beta * C
//	C = alpha * Aᵀ * Bᵀ + beta * C
//
// where A is an m×k or k×m dense matrix, B is an n×k or k×n dense matrix, C is
// an m×n matrix, and alpha and beta are scalars. tA and tB specify whether A or
// B are transposed.
func Gemm(tA, tB blas.Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	switch tA {
	default:
		panic(blas.ErrBadTranspose)
	case blas.NoTrans, blas.Trans, blas.ConjTrans:
	}
	switch tB {
	default:
		panic(blas.ErrBadTranspose)
	case blas.NoTrans, blas.Trans, blas.ConjTrans:
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	aTrans := tA == blas.Trans || tA == blas.ConjTrans
	if aTrans {
		if lda < max(1, m) {
			panic(blas.ErrBadLdA)
		}
	} else {
		if lda < max(1, k) {
			panic(blas.ErrBadLdA)
		}
	}
	bTrans := tB == blas.Trans || tB == blas.ConjTrans
	if bTrans {
		if ldb < max(1, k) {
			panic(blas.ErrBadLdB)
		}
	} else {
		if ldb < max(1, n) {
			panic(blas.ErrBadLdB)
		}
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if aTrans {
		if len(a) < (k-1)*lda+m {
			panic(blas.ErrShortA)
		}
	} else {
		if len(a) < (m-1)*lda+k {
			panic(blas.ErrShortA)
		}
	}
	if bTrans {
		if len(b) < (n-1)*ldb+k {
			panic(blas.ErrShortB)
		}
	} else {
		if len(b) < (k-1)*ldb+n {
			panic(blas.ErrShortB)
		}
	}
	if len(c) < (m-1)*ldc+n {
		panic(blas.ErrShortC)
	}

	// Quick return if possible.
	if (alpha == 0 || k == 0) && beta == 1 {
		return
	}

	// scale c
	if beta != 1 {
		if beta == 0 {
			for i := 0; i < m; i++ {
				ctmp := c[i*ldc : i*ldc+n]
				for j := range ctmp {
					ctmp[j] = 0
				}
			}
		} else {
			for i := 0; i < m; i++ {
				ctmp := c[i*ldc : i*ldc+n]
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
		}
	}

	dgemmParallel(aTrans, bTrans, m, n, k, a, lda, b, ldb, c, ldc, alpha)
}

func dgemmParallel(aTrans, bTrans bool, m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	// dgemmParallel computes a parallel matrix multiplication by partitioning
	// a and b into sub-blocks, and updating c with the multiplication of the sub-block
	// In all cases,
	// A = [ 	A_11	A_12 ... 	A_1j
	//			A_21	A_22 ...	A_2j
	//				...
	//			A_i1	A_i2 ...	A_ij]
	//
	// and same for B. All of the submatrix sizes are blockSize×blockSize except
	// at the edges.
	//
	// In all cases, there is one dimension for each matrix along which
	// C must be updated sequentially.
	// Cij = \sum_k Aik Bki,	(A * B)
	// Cij = \sum_k Aki Bkj,	(Aᵀ * B)
	// Cij = \sum_k Aik Bjk,	(A * Bᵀ)
	// Cij = \sum_k Aki Bjk,	(Aᵀ * Bᵀ)
	//
	// This code computes one {i, j} block sequentially along the k dimension,
	// and computes all of the {i, j} blocks concurrently. This
	// partitioning allows Cij to be updated in-place without race-conditions.
	// Instead of launching a goroutine for each possible concurrent computation,
	// a number of worker goroutines are created and channels are used to pass
	// available and completed cases.
	//
	// http://alexkr.com/docs/matrixmult.pdf is a good reference on matrix-matrix
	// multiplies, though this code does not copy matrices to attempt to eliminate
	// cache misses.

	maxKLen := k
	parBlocks := blas.Blocks(m, blas.BlockSize) * blas.Blocks(n, blas.BlockSize)
	if parBlocks < blas.MinParBlock {
		// The matrix multiplication is small in the dimensions where it can be
		// computed concurrently. Just do it in serial.
		dgemmSerial(aTrans, bTrans, m, n, k, a, lda, b, ldb, c, ldc, alpha)
		return
	}

	// workerLimit acts a number of maximum concurrent workers,
	// with the limit set to the number of procs available.
	workerLimit := make(chan struct{}, runtime.GOMAXPROCS(0))

	// wg is used to wait for all
	var wg sync.WaitGroup
	wg.Add(parBlocks)
	defer wg.Wait()

	for i := 0; i < m; i += blas.BlockSize {
		for j := 0; j < n; j += blas.BlockSize {
			workerLimit <- struct{}{}
			go func(i, j int) {
				defer func() {
					wg.Done()
					<-workerLimit
				}()

				leni := blas.BlockSize
				if i+leni > m {
					leni = m - i
				}
				lenj := blas.BlockSize
				if j+lenj > n {
					lenj = n - j
				}

				cSub := sliceView64(c, ldc, i, j, leni, lenj)

				// Compute A_ik B_kj for all k
				for k := 0; k < maxKLen; k += blas.BlockSize {
					lenk := blas.BlockSize
					if k+lenk > maxKLen {
						lenk = maxKLen - k
					}
					var aSub, bSub []float32
					if aTrans {
						aSub = sliceView64(a, lda, k, i, lenk, leni)
					} else {
						aSub = sliceView64(a, lda, i, k, leni, lenk)
					}
					if bTrans {
						bSub = sliceView64(b, ldb, j, k, lenj, lenk)
					} else {
						bSub = sliceView64(b, ldb, k, j, lenk, lenj)
					}
					dgemmSerial(aTrans, bTrans, leni, lenj, lenk, aSub, lda, bSub, ldb, cSub, ldc, alpha)
				}
			}(i, j)
		}
	}
}

// dgemmSerial is serial matrix multiply
func dgemmSerial(aTrans, bTrans bool, m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	switch {
	case !aTrans && !bTrans:
		dgemmSerialNotNot(m, n, k, a, lda, b, ldb, c, ldc, alpha)
		return
	case aTrans && !bTrans:
		dgemmSerialTransNot(m, n, k, a, lda, b, ldb, c, ldc, alpha)
		return
	case !aTrans && bTrans:
		dgemmSerialNotTrans(m, n, k, a, lda, b, ldb, c, ldc, alpha)
		return
	case aTrans && bTrans:
		dgemmSerialTransTrans(m, n, k, a, lda, b, ldb, c, ldc, alpha)
		return
	default:
		panic("unreachable")
	}
}

// dgemmSerial where neither a nor b are transposed
func dgemmSerialNotNot(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for i := 0; i < m; i++ {
		ctmp := c[i*ldc : i*ldc+n]
		for l, v := range a[i*lda : i*lda+k] {
			tmp := alpha * v
			if tmp != 0 {
				f32.AxpyUnitary(tmp, b[l*ldb:l*ldb+n], ctmp)
			}
		}
	}
}

// dgemmSerial where neither a is transposed and b is not
func dgemmSerialTransNot(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for l := 0; l < k; l++ {
		btmp := b[l*ldb : l*ldb+n]
		for i, v := range a[l*lda : l*lda+m] {
			tmp := alpha * v
			if tmp != 0 {
				ctmp := c[i*ldc : i*ldc+n]
				f32.AxpyUnitary(tmp, btmp, ctmp)
			}
		}
	}
}

// dgemmSerial where neither a is not transposed and b is
func dgemmSerialNotTrans(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for i := 0; i < m; i++ {
		atmp := a[i*lda : i*lda+k]
		ctmp := c[i*ldc : i*ldc+n]
		for j := 0; j < n; j++ {
			ctmp[j] += alpha * f32.DotUnitary(atmp, b[j*ldb:j*ldb+k])
		}
	}
}

// dgemmSerial where both are transposed
func dgemmSerialTransTrans(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, alpha float32) {
	// This style is used instead of the literal [i*stride +j]) is used because
	// approximately 5 times faster as of go 1.3.
	for l := 0; l < k; l++ {
		for i, v := range a[l*lda : l*lda+m] {
			tmp := alpha * v
			if tmp != 0 {
				ctmp := c[i*ldc : i*ldc+n]
				f32.AxpyInc(tmp, b[l:], ctmp, uintptr(n), uintptr(ldb), 1, 0, 0)
			}
		}
	}
}

func sliceView64(a []float32, lda, i, j, r, c int) []float32 {
	return a[i*lda+j : (i+r-1)*lda+j+c]
}

// Symm performs one of the matrix-matrix operations
//
//	C = alpha * A * B + beta * C  if side == blas.Left
//	C = alpha * B * A + beta * C  if side == blas.Right
//
// where A is an n×n or m×m symmetric matrix, B and C are m×n matrices, and alpha
// is a scalar.
func Symm(s blas.Side, ul blas.Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if s != blas.Right && s != blas.Left {
		panic(blas.ErrBadSide)
	}
	if ul != blas.Lower && ul != blas.Upper {
		panic(blas.ErrBadUplo)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	k := n
	if s == blas.Left {
		k = m
	}
	if lda < max(1, k) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, n) {
		panic(blas.ErrBadLdB)
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(blas.ErrShortB)
	}
	if len(c) < ldc*(m-1)+n {
		panic(blas.ErrShortC)
	}

	// Quick return if possible.
	if alpha == 0 && beta == 1 {
		return
	}

	if beta == 0 {
		for i := 0; i < m; i++ {
			ctmp := c[i*ldc : i*ldc+n]
			for j := range ctmp {
				ctmp[j] = 0
			}
		}
	}

	if alpha == 0 {
		if beta != 0 {
			for i := 0; i < m; i++ {
				ctmp := c[i*ldc : i*ldc+n]
				for j := 0; j < n; j++ {
					ctmp[j] *= beta
				}
			}
		}
		return
	}

	isUpper := ul == blas.Upper
	if s == blas.Left {
		for i := 0; i < m; i++ {
			atmp := alpha * a[i*lda+i]
			btmp := b[i*ldb : i*ldb+n]
			ctmp := c[i*ldc : i*ldc+n]
			for j, v := range btmp {
				ctmp[j] *= beta
				ctmp[j] += atmp * v
			}

			for k := 0; k < i; k++ {
				var atmp float32
				if isUpper {
					atmp = a[k*lda+i]
				} else {
					atmp = a[i*lda+k]
				}
				atmp *= alpha
				f32.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ctmp)
			}
			for k := i + 1; k < m; k++ {
				var atmp float32
				if isUpper {
					atmp = a[i*lda+k]
				} else {
					atmp = a[k*lda+i]
				}
				atmp *= alpha
				f32.AxpyUnitary(atmp, b[k*ldb:k*ldb+n], ctmp)
			}
		}
		return
	}
	if isUpper {
		for i := 0; i < m; i++ {
			for j := n - 1; j >= 0; j-- {
				tmp := alpha * b[i*ldb+j]
				var tmp2 float32
				atmp := a[j*lda+j+1 : j*lda+n]
				btmp := b[i*ldb+j+1 : i*ldb+n]
				ctmp := c[i*ldc+j+1 : i*ldc+n]
				for k, v := range atmp {
					ctmp[k] += tmp * v
					tmp2 += btmp[k] * v
				}
				c[i*ldc+j] *= beta
				c[i*ldc+j] += tmp*a[j*lda+j] + alpha*tmp2
			}
		}
		return
	}
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			tmp := alpha * b[i*ldb+j]
			var tmp2 float32
			atmp := a[j*lda : j*lda+j]
			btmp := b[i*ldb : i*ldb+j]
			ctmp := c[i*ldc : i*ldc+j]
			for k, v := range atmp {
				ctmp[k] += tmp * v
				tmp2 += btmp[k] * v
			}
			c[i*ldc+j] *= beta
			c[i*ldc+j] += tmp*a[j*lda+j] + alpha*tmp2
		}
	}
}

// Trmm performs one of the matrix-matrix operations
//
//	B = alpha * A * B   if tA == blas.NoTrans and side == blas.Left
//	B = alpha * Aᵀ * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//	B = alpha * B * A   if tA == blas.NoTrans and side == blas.Right
//	B = alpha * B * Aᵀ  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
//
// where A is an n×n or m×m triangular matrix, B is an m×n matrix, and alpha is a scalar.
func Trmm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	if s != blas.Left && s != blas.Right {
		panic(blas.ErrBadSide)
	}
	if ul != blas.Lower && ul != blas.Upper {
		panic(blas.ErrBadUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(blas.ErrBadTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(blas.ErrBadDiag)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	k := n
	if s == blas.Left {
		k = m
	}
	if lda < max(1, k) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, n) {
		panic(blas.ErrBadLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(blas.ErrShortB)
	}

	if alpha == 0 {
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			for j := range btmp {
				btmp[j] = 0
			}
		}
		return
	}

	nonUnit := d == blas.NonUnit
	if s == blas.Left {
		if tA == blas.NoTrans {
			if ul == blas.Upper {
				for i := 0; i < m; i++ {
					tmp := alpha
					if nonUnit {
						tmp *= a[i*lda+i]
					}
					btmp := b[i*ldb : i*ldb+n]
					f32.ScalUnitary(tmp, btmp)
					for ka, va := range a[i*lda+i+1 : i*lda+m] {
						k := ka + i + 1
						if va != 0 {
							f32.AxpyUnitary(alpha*va, b[k*ldb:k*ldb+n], btmp)
						}
					}
				}
				return
			}
			for i := m - 1; i >= 0; i-- {
				tmp := alpha
				if nonUnit {
					tmp *= a[i*lda+i]
				}
				btmp := b[i*ldb : i*ldb+n]
				f32.ScalUnitary(tmp, btmp)
				for k, va := range a[i*lda : i*lda+i] {
					if va != 0 {
						f32.AxpyUnitary(alpha*va, b[k*ldb:k*ldb+n], btmp)
					}
				}
			}
			return
		}
		// Cases where a is transposed.
		if ul == blas.Upper {
			for k := m - 1; k >= 0; k-- {
				btmpk := b[k*ldb : k*ldb+n]
				for ia, va := range a[k*lda+k+1 : k*lda+m] {
					i := ia + k + 1
					btmp := b[i*ldb : i*ldb+n]
					if va != 0 {
						f32.AxpyUnitary(alpha*va, btmpk, btmp)
					}
				}
				tmp := alpha
				if nonUnit {
					tmp *= a[k*lda+k]
				}
				if tmp != 1 {
					f32.ScalUnitary(tmp, btmpk)
				}
			}
			return
		}
		for k := 0; k < m; k++ {
			btmpk := b[k*ldb : k*ldb+n]
			for i, va := range a[k*lda : k*lda+k] {
				btmp := b[i*ldb : i*ldb+n]
				if va != 0 {
					f32.AxpyUnitary(alpha*va, btmpk, btmp)
				}
			}
			tmp := alpha
			if nonUnit {
				tmp *= a[k*lda+k]
			}
			if tmp != 1 {
				f32.ScalUnitary(tmp, btmpk)
			}
		}
		return
	}
	// Cases where a is on the right
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := 0; i < m; i++ {
				btmp := b[i*ldb : i*ldb+n]
				for k := n - 1; k >= 0; k-- {
					tmp := alpha * btmp[k]
					if tmp == 0 {
						continue
					}
					btmp[k] = tmp
					if nonUnit {
						btmp[k] *= a[k*lda+k]
					}
					f32.AxpyUnitary(tmp, a[k*lda+k+1:k*lda+n], btmp[k+1:n])
				}
			}
			return
		}
		for i := range m {
			btmp := b[i*ldb : i*ldb+n]
			for k := range n {
				tmp := alpha * btmp[k]
				if tmp == 0 {
					continue
				}
				btmp[k] = tmp
				if nonUnit {
					btmp[k] *= a[k*lda+k]
				}
				f32.AxpyUnitary(tmp, a[k*lda:k*lda+k], btmp[:k])
			}
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		for i := range m {
			btmp := b[i*ldb : i*ldb+n]
			for j, vb := range btmp {
				tmp := vb
				if nonUnit {
					tmp *= a[j*lda+j]
				}
				tmp += f32.DotUnitary(a[j*lda+j+1:j*lda+n], btmp[j+1:n])
				btmp[j] = alpha * tmp
			}
		}
		return
	}
	for i := range m {
		btmp := b[i*ldb : i*ldb+n]
		for j := n - 1; j >= 0; j-- {
			tmp := btmp[j]
			if nonUnit {
				tmp *= a[j*lda+j]
			}
			tmp += f32.DotUnitary(a[j*lda:j*lda+j], btmp[:j])
			btmp[j] = alpha * tmp
		}
	}
}

// Trsm solves one of the matrix equations
//
//	A * X = alpha * B   if tA == blas.NoTrans and side == blas.Left
//	Aᵀ * X = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Left
//	X * A = alpha * B   if tA == blas.NoTrans and side == blas.Right
//	X * Aᵀ = alpha * B  if tA == blas.Trans or blas.ConjTrans, and side == blas.Right
//
// where A is an n×n or m×m triangular matrix, X and B are m×n matrices, and alpha is a
// scalar.
//
// At entry to the function, X contains the values of B, and the result is
// stored in-place into X.
//
// No check is made that A is invertible.
func Trsm(s blas.Side, ul blas.Uplo, tA blas.Transpose, d blas.Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int) {
	if s != blas.Left && s != blas.Right {
		panic(blas.ErrBadSide)
	}
	if ul != blas.Lower && ul != blas.Upper {
		panic(blas.ErrBadUplo)
	}
	if tA != blas.NoTrans && tA != blas.Trans && tA != blas.ConjTrans {
		panic(blas.ErrBadTranspose)
	}
	if d != blas.NonUnit && d != blas.Unit {
		panic(blas.ErrBadDiag)
	}
	if m < 0 {
		panic(blas.ErrMLT0)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	k := n
	if s == blas.Left {
		k = m
	}
	if lda < max(1, k) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, n) {
		panic(blas.ErrBadLdB)
	}

	// Quick return if possible.
	if m == 0 || n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(k-1)+k {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(m-1)+n {
		panic(blas.ErrShortB)
	}

	if alpha == 0 {
		for i := 0; i < m; i++ {
			btmp := b[i*ldb : i*ldb+n]
			for j := range btmp {
				btmp[j] = 0
			}
		}
		return
	}
	nonUnit := d == blas.NonUnit
	if s == blas.Left {
		if tA == blas.NoTrans {
			if ul == blas.Upper {
				for i := m - 1; i >= 0; i-- {
					btmp := b[i*ldb : i*ldb+n]
					if alpha != 1 {
						f32.ScalUnitary(alpha, btmp)
					}
					for ka, va := range a[i*lda+i+1 : i*lda+m] {
						if va != 0 {
							k := ka + i + 1
							f32.AxpyUnitary(-va, b[k*ldb:k*ldb+n], btmp)
						}
					}
					if nonUnit {
						tmp := 1 / a[i*lda+i]
						f32.ScalUnitary(tmp, btmp)
					}
				}
				return
			}
			for i := range m {
				btmp := b[i*ldb : i*ldb+n]
				if alpha != 1 {
					f32.ScalUnitary(alpha, btmp)
				}
				for k, va := range a[i*lda : i*lda+i] {
					if va != 0 {
						f32.AxpyUnitary(-va, b[k*ldb:k*ldb+n], btmp)
					}
				}
				if nonUnit {
					tmp := 1 / a[i*lda+i]
					f32.ScalUnitary(tmp, btmp)
				}
			}
			return
		}
		// Cases where a is transposed
		if ul == blas.Upper {
			for k := range m {
				btmpk := b[k*ldb : k*ldb+n]
				if nonUnit {
					tmp := 1 / a[k*lda+k]
					f32.ScalUnitary(tmp, btmpk)
				}
				for ia, va := range a[k*lda+k+1 : k*lda+m] {
					if va != 0 {
						i := ia + k + 1
						f32.AxpyUnitary(-va, btmpk, b[i*ldb:i*ldb+n])
					}
				}
				if alpha != 1 {
					f32.ScalUnitary(alpha, btmpk)
				}
			}
			return
		}
		for k := m - 1; k >= 0; k-- {
			btmpk := b[k*ldb : k*ldb+n]
			if nonUnit {
				tmp := 1 / a[k*lda+k]
				f32.ScalUnitary(tmp, btmpk)
			}
			for i, va := range a[k*lda : k*lda+k] {
				if va != 0 {
					f32.AxpyUnitary(-va, btmpk, b[i*ldb:i*ldb+n])
				}
			}
			if alpha != 1 {
				f32.ScalUnitary(alpha, btmpk)
			}
		}
		return
	}
	// Cases where a is to the right of X.
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := range m {
				btmp := b[i*ldb : i*ldb+n]
				if alpha != 1 {
					f32.ScalUnitary(alpha, btmp)
				}
				for k, vb := range btmp {
					if vb == 0 {
						continue
					}
					if nonUnit {
						btmp[k] /= a[k*lda+k]
					}
					f32.AxpyUnitary(-btmp[k], a[k*lda+k+1:k*lda+n], btmp[k+1:n])
				}
			}
			return
		}
		for i := range m {
			btmp := b[i*ldb : i*ldb+n]
			if alpha != 1 {
				f32.ScalUnitary(alpha, btmp)
			}
			for k := n - 1; k >= 0; k-- {
				if btmp[k] == 0 {
					continue
				}
				if nonUnit {
					btmp[k] /= a[k*lda+k]
				}
				f32.AxpyUnitary(-btmp[k], a[k*lda:k*lda+k], btmp[:k])
			}
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		for i := range m {
			btmp := b[i*ldb : i*ldb+n]
			for j := n - 1; j >= 0; j-- {
				tmp := alpha*btmp[j] - f32.DotUnitary(a[j*lda+j+1:j*lda+n], btmp[j+1:])
				if nonUnit {
					tmp /= a[j*lda+j]
				}
				btmp[j] = tmp
			}
		}
		return
	}
	for i := range m {
		btmp := b[i*ldb : i*ldb+n]
		for j := range n {
			tmp := alpha*btmp[j] - f32.DotUnitary(a[j*lda:j*lda+j], btmp[:j])
			if nonUnit {
				tmp /= a[j*lda+j]
			}
			btmp[j] = tmp
		}
	}
}

// Syrk performs one of the symmetric rank-k operations
//
//	C = alpha * A * Aᵀ + beta * C  if tA == blas.NoTrans
//	C = alpha * Aᵀ * A + beta * C  if tA == blas.Trans or tA == blas.ConjTrans
//
// where A is an n×k or k×n matrix, C is an n×n symmetric matrix, and alpha and
// beta are scalars.
func Syrk(ul blas.Uplo, tA blas.Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(blas.ErrBadUplo)
	}
	if tA != blas.Trans && tA != blas.NoTrans && tA != blas.ConjTrans {
		panic(blas.ErrBadTranspose)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	row, col := k, n
	if tA == blas.NoTrans {
		row, col = n, k
	}
	if lda < max(1, col) {
		panic(blas.ErrBadLdA)
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(row-1)+col {
		panic(blas.ErrShortA)
	}
	if len(c) < ldc*(n-1)+n {
		panic(blas.ErrShortC)
	}

	if alpha == 0 {
		if beta == 0 {
			if ul == blas.Upper {
				for i := 0; i < n; i++ {
					ctmp := c[i*ldc+i : i*ldc+n]
					for j := range ctmp {
						ctmp[j] = 0
					}
				}
				return
			}
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc : i*ldc+i+1]
				for j := range ctmp {
					ctmp[j] = 0
				}
			}
			return
		}
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc+i : i*ldc+n]
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc : i*ldc+i+1]
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		return
	}
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc+i : i*ldc+n]
				atmp := a[i*lda : i*lda+k]
				if beta == 0 {
					for jc := range ctmp {
						j := jc + i
						ctmp[jc] = alpha * f32.DotUnitary(atmp, a[j*lda:j*lda+k])
					}
				} else {
					for jc, vc := range ctmp {
						j := jc + i
						ctmp[jc] = vc*beta + alpha*f32.DotUnitary(atmp, a[j*lda:j*lda+k])
					}
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc : i*ldc+i+1]
			atmp := a[i*lda : i*lda+k]
			if beta == 0 {
				for j := range ctmp {
					ctmp[j] = alpha * f32.DotUnitary(a[j*lda:j*lda+k], atmp)
				}
			} else {
				for j, vc := range ctmp {
					ctmp[j] = vc*beta + alpha*f32.DotUnitary(a[j*lda:j*lda+k], atmp)
				}
			}
		}
		return
	}
	// Cases where a is transposed.
	if ul == blas.Upper {
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc+i : i*ldc+n]
			if beta == 0 {
				for j := range ctmp {
					ctmp[j] = 0
				}
			} else if beta != 1 {
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			for l := 0; l < k; l++ {
				tmp := alpha * a[l*lda+i]
				if tmp != 0 {
					f32.AxpyUnitary(tmp, a[l*lda+i:l*lda+n], ctmp)
				}
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		ctmp := c[i*ldc : i*ldc+i+1]
		if beta != 1 {
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		for l := 0; l < k; l++ {
			tmp := alpha * a[l*lda+i]
			if tmp != 0 {
				f32.AxpyUnitary(tmp, a[l*lda:l*lda+i+1], ctmp)
			}
		}
	}
}

// Syr2k performs one of the symmetric rank 2k operations
//
//	C = alpha * A * Bᵀ + alpha * B * Aᵀ + beta * C  if tA == blas.NoTrans
//	C = alpha * Aᵀ * B + alpha * Bᵀ * A + beta * C  if tA == blas.Trans or tA == blas.ConjTrans
//
// where A and B are n×k or k×n matrices, C is an n×n symmetric matrix, and
// alpha and beta are scalars.
func Syr2k(ul blas.Uplo, tA blas.Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int) {
	if ul != blas.Lower && ul != blas.Upper {
		panic(blas.ErrBadUplo)
	}
	if tA != blas.Trans && tA != blas.NoTrans && tA != blas.ConjTrans {
		panic(blas.ErrBadTranspose)
	}
	if n < 0 {
		panic(blas.ErrNLT0)
	}
	if k < 0 {
		panic(blas.ErrKLT0)
	}
	row, col := k, n
	if tA == blas.NoTrans {
		row, col = n, k
	}
	if lda < max(1, col) {
		panic(blas.ErrBadLdA)
	}
	if ldb < max(1, col) {
		panic(blas.ErrBadLdB)
	}
	if ldc < max(1, n) {
		panic(blas.ErrBadLdC)
	}

	// Quick return if possible.
	if n == 0 {
		return
	}

	// For zero matrix size the following slice length checks are trivially satisfied.
	if len(a) < lda*(row-1)+col {
		panic(blas.ErrShortA)
	}
	if len(b) < ldb*(row-1)+col {
		panic(blas.ErrShortB)
	}
	if len(c) < ldc*(n-1)+n {
		panic(blas.ErrShortC)
	}

	if alpha == 0 {
		if beta == 0 {
			if ul == blas.Upper {
				for i := 0; i < n; i++ {
					ctmp := c[i*ldc+i : i*ldc+n]
					for j := range ctmp {
						ctmp[j] = 0
					}
				}
				return
			}
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc : i*ldc+i+1]
				for j := range ctmp {
					ctmp[j] = 0
				}
			}
			return
		}
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				ctmp := c[i*ldc+i : i*ldc+n]
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc : i*ldc+i+1]
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		return
	}
	if tA == blas.NoTrans {
		if ul == blas.Upper {
			for i := 0; i < n; i++ {
				atmp := a[i*lda : i*lda+k]
				btmp := b[i*ldb : i*ldb+k]
				ctmp := c[i*ldc+i : i*ldc+n]
				if beta == 0 {
					for jc := range ctmp {
						j := i + jc
						var tmp1, tmp2 float32
						binner := b[j*ldb : j*ldb+k]
						for l, v := range a[j*lda : j*lda+k] {
							tmp1 += v * btmp[l]
							tmp2 += atmp[l] * binner[l]
						}
						ctmp[jc] = alpha * (tmp1 + tmp2)
					}
				} else {
					for jc := range ctmp {
						j := i + jc
						var tmp1, tmp2 float32
						binner := b[j*ldb : j*ldb+k]
						for l, v := range a[j*lda : j*lda+k] {
							tmp1 += v * btmp[l]
							tmp2 += atmp[l] * binner[l]
						}
						ctmp[jc] *= beta
						ctmp[jc] += alpha * (tmp1 + tmp2)
					}
				}
			}
			return
		}
		for i := 0; i < n; i++ {
			atmp := a[i*lda : i*lda+k]
			btmp := b[i*ldb : i*ldb+k]
			ctmp := c[i*ldc : i*ldc+i+1]
			if beta == 0 {
				for j := 0; j <= i; j++ {
					var tmp1, tmp2 float32
					binner := b[j*ldb : j*ldb+k]
					for l, v := range a[j*lda : j*lda+k] {
						tmp1 += v * btmp[l]
						tmp2 += atmp[l] * binner[l]
					}
					ctmp[j] = alpha * (tmp1 + tmp2)
				}
			} else {
				for j := 0; j <= i; j++ {
					var tmp1, tmp2 float32
					binner := b[j*ldb : j*ldb+k]
					for l, v := range a[j*lda : j*lda+k] {
						tmp1 += v * btmp[l]
						tmp2 += atmp[l] * binner[l]
					}
					ctmp[j] *= beta
					ctmp[j] += alpha * (tmp1 + tmp2)
				}
			}
		}
		return
	}
	if ul == blas.Upper {
		for i := 0; i < n; i++ {
			ctmp := c[i*ldc+i : i*ldc+n]
			switch beta {
			case 0:
				for j := range ctmp {
					ctmp[j] = 0
				}
			case 1:
			default:
				for j := range ctmp {
					ctmp[j] *= beta
				}
			}
			for l := 0; l < k; l++ {
				tmp1 := alpha * b[l*ldb+i]
				tmp2 := alpha * a[l*lda+i]
				btmp := b[l*ldb+i : l*ldb+n]
				if tmp1 != 0 || tmp2 != 0 {
					for j, v := range a[l*lda+i : l*lda+n] {
						ctmp[j] += v*tmp1 + btmp[j]*tmp2
					}
				}
			}
		}
		return
	}
	for i := 0; i < n; i++ {
		ctmp := c[i*ldc : i*ldc+i+1]
		switch beta {
		case 0:
			for j := range ctmp {
				ctmp[j] = 0
			}
		case 1:
		default:
			for j := range ctmp {
				ctmp[j] *= beta
			}
		}
		for l := 0; l < k; l++ {
			tmp1 := alpha * b[l*ldb+i]
			tmp2 := alpha * a[l*lda+i]
			btmp := b[l*ldb : l*ldb+i+1]
			if tmp1 != 0 || tmp2 != 0 {
				for j, v := range a[l*lda : l*lda+i+1] {
					ctmp[j] += v*tmp1 + btmp[j]*tmp2
				}
			}
		}
	}
}
