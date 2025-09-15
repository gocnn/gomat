package blas

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
	Saxpy(n int, alpha float32, x []float32, incX int, y []float32, incY int)
	Sscal(n int, alpha float32, x []float32, incX int)
	Scopy(n int, x []float32, incX int, y []float32, incY int)
	Sswap(n int, x []float32, incX int, y []float32, incY int)

	Sdot(n int, x []float32, incX int, y []float32, incY int) float32
	Sdsdot(n int, alpha float32, x []float32, incX int, y []float32, incY int) float32

	Snrm2(n int, x []float32, incX int) float32
	Sasum(n int, x []float32, incX int) float32
	Isamax(n int, x []float32, incX int) int

	Srotg(a, b float32) (c, s, r, z float32)
	Srot(n int, x []float32, incX int, y []float32, incY int, c, s float32)
	Srotmg(d1, d2, b1, b2 float32) (p SrotmParams, rd1, rd2, rb1 float32)
	Srotm(n int, x []float32, incX int, y []float32, incY int, p SrotmParams)
}

// Float32Level2 implements the single precision real BLAS Level 2 routines.
type Float32Level2 interface {
	Sgemv(tA Transpose, m, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Ssymv(ul Uplo, n int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Strmv(ul Uplo, tA Transpose, d Diag, n int, a []float32, lda int, x []float32, incX int)
	Strsv(ul Uplo, tA Transpose, d Diag, n int, a []float32, lda int, x []float32, incX int)

	Sger(m, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int)

	Ssyr(ul Uplo, n int, alpha float32, x []float32, incX int, a []float32, lda int)
	Ssyr2(ul Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32, lda int)

	Sgbmv(tA Transpose, m, n, kL, kU int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Ssbmv(ul Uplo, n, k int, alpha float32, a []float32, lda int, x []float32, incX int, beta float32, y []float32, incY int)
	Stbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []float32, lda int, x []float32, incX int)
	Stbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []float32, lda int, x []float32, incX int)

	Sspmv(ul Uplo, n int, alpha float32, ap []float32, x []float32, incX int, beta float32, y []float32, incY int)
	Stpmv(ul Uplo, tA Transpose, d Diag, n int, ap []float32, x []float32, incX int)
	Stpsv(ul Uplo, tA Transpose, d Diag, n int, ap []float32, x []float32, incX int)

	Sspr(ul Uplo, n int, alpha float32, x []float32, incX int, ap []float32)
	Sspr2(ul Uplo, n int, alpha float32, x []float32, incX int, y []float32, incY int, a []float32)
}

// Float32Level3 implements the single precision real BLAS Level 3 routines.
type Float32Level3 interface {
	Sgemm(tA, tB Transpose, m, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	Ssymm(s Side, ul Uplo, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
	Strmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)
	Strsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float32, a []float32, lda int, b []float32, ldb int)

	Ssyrk(ul Uplo, t Transpose, n, k int, alpha float32, a []float32, lda int, beta float32, c []float32, ldc int)
	Ssyr2k(ul Uplo, t Transpose, n, k int, alpha float32, a []float32, lda int, b []float32, ldb int, beta float32, c []float32, ldc int)
}

// Float64 implements the single precision real BLAS routines.
type Float64 interface {
	Float64Level1
	Float64Level2
	Float64Level3
}

// Float64Level1 implements the double precision real BLAS Level 1 routines.
type Float64Level1 interface {
	Daxpy(n int, alpha float64, x []float64, incX int, y []float64, incY int)
	Dscal(n int, alpha float64, x []float64, incX int)
	Dcopy(n int, x []float64, incX int, y []float64, incY int)
	Dswap(n int, x []float64, incX int, y []float64, incY int)

	Ddot(n int, x []float64, incX int, y []float64, incY int) float64

	Dnrm2(n int, x []float64, incX int) float64
	Dasum(n int, x []float64, incX int) float64
	Idamax(n int, x []float64, incX int) int

	Drotg(a, b float64) (c, s, r, z float64)
	Drot(n int, x []float64, incX int, y []float64, incY int, c float64, s float64)
	Drotmg(d1, d2, b1, b2 float64) (p DrotmParams, rd1, rd2, rb1 float64)
	Drotm(n int, x []float64, incX int, y []float64, incY int, p DrotmParams)
}

// Float64Level2 implements the double precision real BLAS Level 2 routines.
type Float64Level2 interface {
	Dgemv(tA Transpose, m, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Dsymv(ul Uplo, n int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Dtrmv(ul Uplo, tA Transpose, d Diag, n int, a []float64, lda int, x []float64, incX int)
	Dtrsv(ul Uplo, tA Transpose, d Diag, n int, a []float64, lda int, x []float64, incX int)

	Dger(m, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int)

	Dsyr(ul Uplo, n int, alpha float64, x []float64, incX int, a []float64, lda int)
	Dsyr2(ul Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64, lda int)

	Dgbmv(tA Transpose, m, n, kL, kU int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Dsbmv(ul Uplo, n, k int, alpha float64, a []float64, lda int, x []float64, incX int, beta float64, y []float64, incY int)
	Dtbmv(ul Uplo, tA Transpose, d Diag, n, k int, a []float64, lda int, x []float64, incX int)
	Dtbsv(ul Uplo, tA Transpose, d Diag, n, k int, a []float64, lda int, x []float64, incX int)

	Dspmv(ul Uplo, n int, alpha float64, ap []float64, x []float64, incX int, beta float64, y []float64, incY int)
	Dtpmv(ul Uplo, tA Transpose, d Diag, n int, ap []float64, x []float64, incX int)
	Dtpsv(ul Uplo, tA Transpose, d Diag, n int, ap []float64, x []float64, incX int)

	Dspr(ul Uplo, n int, alpha float64, x []float64, incX int, ap []float64)
	Dspr2(ul Uplo, n int, alpha float64, x []float64, incX int, y []float64, incY int, a []float64)
}

// Float64Level3 implements the double precision real BLAS Level 3 routines.
type Float64Level3 interface {
	Dgemm(tA, tB Transpose, m, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	Dsymm(s Side, ul Uplo, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
	Dtrmm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)
	Dtrsm(s Side, ul Uplo, tA Transpose, d Diag, m, n int, alpha float64, a []float64, lda int, b []float64, ldb int)

	Dsyrk(ul Uplo, t Transpose, n, k int, alpha float64, a []float64, lda int, beta float64, c []float64, ldc int)
	Dsyr2k(ul Uplo, t Transpose, n, k int, alpha float64, a []float64, lda int, b []float64, ldb int, beta float64, c []float64, ldc int)
}
