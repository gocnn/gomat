// Copyright 2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package blas

// Panic strings used during parameter checks.
// This list is duplicated in netlib/blas/netlib. Keep in sync.
const (
	ErrZeroIncX = "blas: zero x index increment"
	ErrZeroIncY = "blas: zero y index increment"

	ErrMLT0  = "blas: m < 0"
	ErrNLT0  = "blas: n < 0"
	ErrKLT0  = "blas: k < 0"
	ErrKLLT0 = "blas: kL < 0"
	ErrKULT0 = "blas: kU < 0"

	ErrBadUplo      = "blas: illegal triangle"
	ErrBadTranspose = "blas: illegal transpose"
	ErrBadDiag      = "blas: illegal diagonal"
	ErrBadSide      = "blas: illegal side"
	ErrBadFlag      = "blas: illegal rotm flag"

	ErrBadLdA = "blas: bad leading dimension of A"
	ErrBadLdB = "blas: bad leading dimension of B"
	ErrBadLdC = "blas: bad leading dimension of C"

	ErrShortX  = "blas: insufficient length of x"
	ErrShortY  = "blas: insufficient length of y"
	ErrShortAP = "blas: insufficient length of ap"
	ErrShortA  = "blas: insufficient length of a"
	ErrShortB  = "blas: insufficient length of b"
	ErrShortC  = "blas: insufficient length of c"
)
