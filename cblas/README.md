# CBLAS

Go bindings to external BLAS libraries for maximum performance.

## Supported Libraries

- **Intel MKL** (Windows, Linux) - High performance math library
- **OpenBLAS** (Linux, Windows) - Open source BLAS implementation  
- **Apple Accelerate** (macOS) - Built-in optimized framework
- **ATLAS** - Automatically Tuned Linear Algebra Software
- Any CBLAS-compatible library

## Installation & Setup

### Windows (Intel MKL - Recommended)

1. Download and install Intel oneAPI Base Toolkit from:
   <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html>

2. The default installation path is: `C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib`

3. No additional configuration needed - the library is automatically detected.

### Linux (OpenBLAS)

1. Install OpenBLAS following the guide at:
   <http://www.openmathlib.org/OpenBLAS/docs/install>

2. Common installation methods:

    ```bash
    # Ubuntu/Debian
    sudo apt-get install libopenblas-dev

    # CentOS/RHEL/Fedora
    sudo dnf install openblas-devel

    # OpenSUSE/SLE
    sudo zypper install openblas-devel

    # Arch/Manjaro/Antergos
    sudo pacman -S openblas
    ```

3. No additional configuration needed - the library is automatically detected.

### macOS (Apple Accelerate)

No installation required! Apple Accelerate framework is built into macOS and provides optimized BLAS routines, especially for Apple Silicon processors.

## Custom Configuration

Override the default library with environment variables:

```bash
# For Intel MKL
export CGO_LDFLAGS="-L/path/to/mkl/lib -lmkl_rt"

# For OpenBLAS
export CGO_LDFLAGS="-L/path/to/openblas/lib -lopenblas"

# For system BLAS
export CGO_LDFLAGS="-lblas"
```

## Usage

```go
import "github.com/gocnn/gomat/cblas/cblas64"

// Matrix-vector multiplication: y = αAx + βy
cblas64.Gemv(blas.NoTrans, m, n, α, A, lda, x, incX, β, y, incY)
```

Use `cblas32` for float32 precision.
