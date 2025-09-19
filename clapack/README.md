# CLAPACK

Go bindings to external LAPACK libraries for maximum performance.

## Libraries

Intel MKL, OpenBLAS, ATLAS, or any CLAPACK-compatible library.

## Setup

Set library path before building:

```bash
export CGO_LDFLAGS="-L/path/to/lib -llapack"
```

For Intel MKL:

```bash
export CGO_LDFLAGS="-L/path/to/mkl/lib -lmkl_rt"
```

## Usage

```go
import "github.com/gocnn/gomat/clapack/clapack64"

// LU factorization: A = P*L*U
clapack64.Getrf(m, n, A, lda, ipiv)
```

Use `clapack32` for float32 precision.
