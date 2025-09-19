# gomat

High-performance **BLAS** and **LAPACK** library implemented in Go with assembly optimizations and Intel MKL/OpenBLAS acceleration support. Built upon proven [Gonum](https://www.gonum.org/) algorithms, focused on single-function matrix operators with **row-major storage** (C-style).

Optional external library acceleration: [CBLAS](cblas/README.md) | [CLAPACK](clapack/README.md)

## Install

```bash
go get github.com/gocnn/gomat
```

## License

This BLAS/LAPACK implementation is based on the reference BLAS and LAPACK from `Netlib`, which are in the public domain.

### Netlib License

The original BLAS and LAPACK from `Netlib` are available at:

- See: [http://www.netlib.org/blas/](http://www.netlib.org/blas/)
- LAPACK: [http://www.netlib.org/lapack/](http://www.netlib.org/lapack/)
- BLAS Technical Forum: [http://www.netlib.org/blas/blast-forum/](http://www.netlib.org/blas/blast-forum/)
- The reference BLAS and LAPACK are in the public domain and freely available for use.

### Gonum License

This Go implementation follows the Gonum project licensing:

- Copyright 2015 The Gonum Authors. All rights reserved.
- Use of this source code is governed by a BSD-style license that can be found in the [LICENSE](blas/LICENSE) file.
- See: [https://github.com/gonum/gonum/blob/master/LICENSE](https://github.com/gonum/gonum/blob/master/LICENSE)
