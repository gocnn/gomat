# CBLAS

Go bindings to external BLAS libraries for maximum performance.

## Libraries

Intel MKL, OpenBLAS, ATLAS, or any CBLAS-compatible library.

## Setup

Set library path before building:

```bash
export CGO_LDFLAGS="-L/path/to/lib -lblas"
```

For Intel MKL:

```bash
export CGO_LDFLAGS="-L/path/to/mkl/lib -lmkl_rt"
```

## Usage

```go
import "github.com/gocnn/gomat/cblas/cblas64"

// Matrix-vector multiplication: y = αAx + βy
cblas64.Gemv(blas.NoTrans, m, n, α, A, lda, x, incX, β, y, incY)
```

Use `cblas32` for float32 precision.
