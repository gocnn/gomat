package blas

//go:generate go run generate.go

// [SD]gemm behavior constants. These are kept here to keep them out of the
// way during single precision code generation.
const (
	BlockSize   = 64 // b x b matrix
	MinParBlock = 4  // minimum number of blocks needed to go parallel
)

// Blocks returns the number of divisions of the dimension length with the given
// block size.
func Blocks(dim, bsize int) int {
	return (dim + bsize - 1) / bsize
}

// Flag constants indicate Givens transformation H matrix state.
type Flag int

const (
	Identity    Flag = -2 // H is the identity matrix; no rotation is needed.
	Rescaling   Flag = -1 // H specifies rescaling.
	OffDiagonal Flag = 0  // Off-diagonal elements of H are non-unit.
	Diagonal    Flag = 1  // Diagonal elements of H are non-unit.
)

// SrotmParams contains Givens transformation parameters returned
// by the Float32 Srotm method.
type SrotmParams struct {
	Flag
	H [4]float32 // Column-major 2 by 2 matrix.
}

// DrotmParams contains Givens transformation parameters returned
// by the Float64 Drotm method.
type DrotmParams struct {
	Flag
	H [4]float64 // Column-major 2 by 2 matrix.
}

// Transpose specifies the transposition operation of a matrix.
type Transpose byte

const (
	NoTrans   Transpose = 'N'
	Trans     Transpose = 'T'
	ConjTrans Transpose = 'C'
)

// Uplo specifies whether a matrix is upper or lower triangular.
type Uplo byte

const (
	Upper Uplo = 'U'
	Lower Uplo = 'L'
	All   Uplo = 'A'
)

// Diag specifies whether a matrix is unit triangular.
type Diag byte

const (
	NonUnit Diag = 'N'
	Unit    Diag = 'U'
)

// Side specifies from which side a multiplication operation is performed.
type Side byte

const (
	Left  Side = 'L'
	Right Side = 'R'
)

// Float32 implements the single precision real BLAS routines.
type Float32 interface {
	Float32Level1
	Float32Level2
	Float32Level3
}

// Float32Level1 implements the single precision real BLAS Level 1 routines.
type Float32Level1 interface {
	Axpy(n int, alpha float32, x []float32, incX int, y []float32, incY int)
	Scal(n int, alpha float32, x []float32, incX int)
	Copy(n int, x []float32, incX int, y []float32, incY int)
	Swap(n int, x []float32, incX int, y []float32, incY int)

	Dot(n int, x []float32, incX int, y []float32, incY int) float32
	Dsdot(n int, alpha float32, x []float32, incX int, y []float32, incY int) float32

	Nrm2(n int, x []float32, incX int) float32
	Asum(n int, x []float32, incX int) float32
	Iamax(n int, x []float32, incX int) int

	Rotg(a, b float32) (c, s, r, z float32)
	Rot(n int, x []float32, incX int, y []float32, incY int, c, s float32)
	Rotmg(d1, d2, b1, b2 float32) (p SrotmParams, rd1, rd2, rb1 float32)
	Rotm(n int, x []float32, incX int, y []float32, incY int, p SrotmParams)
}

// Float32Level2 implements the single precision real BLAS Level 2 routines.
type Float32Level2 interface {
	Gemv(tA Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Symv(ul Uplo, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Trmv(ul Uplo, tA Transpose, d Diag, n int, a []float32, lda int, x []float32, incX int)
	Trsv(ul Uplo, tA Transpose, d Diag, n int, a []float32, lda int, x []float32, incX int)

	Ger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int)

	Syr(ul Uplo, n int, alpha float32, x []float32, incX int, a []float32, lda int)
	Syr2(ul Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int)

	Gbmv(tA Transpose, m, n, kL, kU int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Sbmv(ul Uplo, n, k int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Tbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []float32, lda int, x []float32, incX int)
	Tbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []float32, lda int, x []float32, incX int)

	Spmv(ul Uplo, n int, alpha float32, ap []float32, x []float32, incX int, beta float32, y []float32, incY int)
	Tpmv(ul Uplo, tA Transpose, d Diag, n int, ap []float32, x []float32, incX int)
	Tpsv(ul Uplo, tA Transpose, d Diag, n int, ap []float32, x []float32, incX int)

	Spr(ul Uplo, n int, alpha float32, x []float32, incX int, ap []float32)
	Spr2(ul Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32)
}

// Float32Level3 implements the single precision real BLAS Level 3 routines.
type Float32Level3 interface {
	Gemm(tA, tB Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	Symm(s Side, ul Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	Trmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)
	Trsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)

	Syrk(ul Uplo, t Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int)
	Syr2k(ul Uplo, t Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
}

// Float64 implements the double precision real BLAS routines.
type Float64 interface {
	Float64Level1
	Float64Level2
	Float64Level3
}

// Float64Level1 implements the double precision real BLAS Level 1 routines.
type Float64Level1 interface {
	Axpy(n int, alpha float64, x []float64, incX int, y []float64, incY int)
	Scal(n int, alpha float64, x []float64, incX int)
	Copy(n int, x []float64, incX int, y []float64, incY int)
	Swap(n int, x []float64, incX int, y []float64, incY int)

	Dot(n int, x []float64, incX int, y []float64, incY int) float64
	Dsdot(n int, alpha float64, x []float64, incX int, y []float64, incY int) float64

	Nrm2(n int, x []float64, incX int) float64
	Asum(n int, x []float64, incX int) float64
	Iamax(n int, x []float64, incX int) int

	Rotg(a, b float64) (c, s, r, z float64)
	Rot(n int, x []float64, incX int, y []float64, incY int, c, s float64)
	Rotmg(d1, d2, b1, b2 float64) (p DrotmParams, rd1, rd2, rb1 float64)
	Rotm(n int, x []float64, incX int, y []float64, incY int, p DrotmParams)
}

// Float64Level2 implements the double precision real BLAS Level 2 routines.
type Float64Level2 interface {
	Gemv(tA Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Symv(ul Uplo, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Trmv(ul Uplo, tA Transpose, d Diag, n int, a []float64, lda int, x []float64, incX int)
	Trsv(ul Uplo, tA Transpose, d Diag, n int, a []float64, lda int, x []float64, incX int)

	Ger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int)

	Syr(ul Uplo, n int, alpha float64, x []float64, incX int, a []float64, lda int)
	Syr2(ul Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int)

	Gbmv(tA Transpose, m, n, kL, kU int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Sbmv(ul Uplo, n, k int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Tbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []float64, lda int, x []float64, incX int)
	Tbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []float64, lda int, x []float64, incX int)

	Spmv(ul Uplo, n int, alpha float64, ap []float64, x []float64, incX int, beta float64, y []float64, incY int)
	Tpmv(ul Uplo, tA Transpose, d Diag, n int, ap []float64, x []float64, incX int)
	Tpsv(ul Uplo, tA Transpose, d Diag, n int, ap []float64, x []float64, incX int)

	Spr(ul Uplo, n int, alpha float64, x []float64, incX int, ap []float64)
	Spr2(ul Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64)
}

// Float64Level3 implements the double precision real BLAS Level 3 routines.
type Float64Level3 interface {
	Gemm(tA, tB Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	Symm(s Side, ul Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	Trmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)
	Trsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)

	Syrk(ul Uplo, t Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int)
	Syr2k(ul Uplo, t Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
}
