package main

//go:generate go run . -out ../../mat/log_amd64.s -stubs ../../mat/log_amd64.go -pkg mat

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

func main() {
	ConstraintExpr("amd64,gc,!purego")

	buildAVX(32)
	buildSSE(32)

	Generate()
}

func buildAVX(bits int) {
	LCPI0_0 := ConstData("AVX2_LCPI0_0", U32(0x00800000))   // float 1.17549435E-38
	LCPI0_1 := ConstData("AVX2_LCPI0_1", U32(2155872255))   // 0x807fffff
	LCPI0_2 := ConstData("AVX2_LCPI0_2", U32(1056964608))   // 0x3f000000
	LCPI0_3 := ConstData("AVX2_LCPI0_3", U32(4294967169))   // 0xffffff81
	LCPI0_4 := ConstData("AVX2_LCPI0_4", U32(0x3f800000))   // float 1
	LCPI0_5 := ConstData("AVX2_LCPI0_5", U32(0x3f3504f3))   // float 0.707106769
	LCPI0_6 := ConstData("AVX2_LCPI0_6", U32(0xbf800000))   // float -1
	LCPI0_7 := ConstData("AVX2_LCPI0_7", U32(0x3d9021bb))   // float 0.0703768358
	LCPI0_8 := ConstData("AVX2_LCPI0_8", U32(0xbdebd1b8))   // float -0.115146101
	LCPI0_9 := ConstData("AVX2_LCPI0_9", U32(0x3def251a))   // float 0.116769984
	LCPI0_10 := ConstData("AVX2_LCPI0_10", U32(0xbdfe5d4f)) // float -0.12420141
	LCPI0_11 := ConstData("AVX2_LCPI0_11", U32(0x3e11e9bf)) // float 0.142493233
	LCPI0_12 := ConstData("AVX2_LCPI0_12", U32(0xbe2aae50)) // float -0.166680574
	LCPI0_13 := ConstData("AVX2_LCPI0_13", U32(0x3e4cceac)) // float 0.200007141
	LCPI0_14 := ConstData("AVX2_LCPI0_14", U32(0xbe7ffffc)) // float -0.24999994
	LCPI0_15 := ConstData("AVX2_LCPI0_15", U32(0x3eaaaaaa)) // float 0.333333313
	LCPI0_16 := ConstData("AVX2_LCPI0_16", U32(0xb95e8083)) // float -2.12194442E-4
	LCPI0_17 := ConstData("AVX2_LCPI0_17", U32(0xbf000000)) // float -0.5
	LCPI0_18 := ConstData("AVX2_LCPI0_18", U32(0x3f318000)) // float 0.693359375

	name := fmt.Sprintf("LogAVX%d", bits)
	signature := fmt.Sprintf("func(x, y []float%d)", bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf(
		"%s computes the natural logarithm of each element of x, storing the result in y (%d bits, AVX2 required).",
		name, bits,
	))

	x := Mem{Base: Load(Param("x").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}

	VMOVUPS(x, Y0)

	// ---

	//    vxorps  %xmm1, %xmm1, %xmm1
	VXORPS(X1, X1, X1)

	//    vcmpleps        %ymm1, %ymm0, %ymm1
	VCMPPS(U8(2), Y1, Y0, Y1)

	//    vbroadcastss    .LCPI0_0(%rip), %ymm2   # ymm2 = [1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38]
	VBROADCASTSS(LCPI0_0, Y2)

	//    vmaxps  %ymm2, %ymm0, %ymm0
	VMAXPS(Y2, Y0, Y0)

	//    vpsrld  $23, %ymm0, %ymm2
	VPSRLD(U8(23), Y0, Y2)

	//    vbroadcastss    .LCPI0_1(%rip), %ymm3   # ymm3 = [2155872255,2155872255,2155872255,2155872255,2155872255,2155872255,2155872255,2155872255]
	VBROADCASTSS(LCPI0_1, Y3)

	//    vandps  %ymm3, %ymm0, %ymm0
	VANDPS(Y3, Y0, Y0)

	//    vbroadcastss    .LCPI0_2(%rip), %ymm3   # ymm3 = [1056964608,1056964608,1056964608,1056964608,1056964608,1056964608,1056964608,1056964608]
	VBROADCASTSS(LCPI0_2, Y3)

	//    vpbroadcastd    .LCPI0_3(%rip), %ymm4   # ymm4 = [4294967169,4294967169,4294967169,4294967169,4294967169,4294967169,4294967169,4294967169]
	VPBROADCASTD(LCPI0_3, Y4)

	//    vorps   %ymm3, %ymm0, %ymm0
	VORPS(Y3, Y0, Y0)

	//    vpaddd  %ymm4, %ymm2, %ymm2
	VPADDD(Y4, Y2, Y2)

	//    vcvtdq2ps       %ymm2, %ymm2
	VCVTDQ2PS(Y2, Y2)

	//    vbroadcastss    .LCPI0_4(%rip), %ymm3   # ymm3 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	VBROADCASTSS(LCPI0_4, Y3)

	//    vaddps  %ymm3, %ymm2, %ymm2
	VADDPS(Y3, Y2, Y2)

	//    vbroadcastss    .LCPI0_5(%rip), %ymm4   # ymm4 = [7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1]
	VBROADCASTSS(LCPI0_5, Y4)

	//    vcmpltps        %ymm4, %ymm0, %ymm4
	VCMPPS(U8(1), Y4, Y0, Y4)

	//    vandps  %ymm0, %ymm4, %ymm5
	VANDPS(Y0, Y4, Y5)

	//    vbroadcastss    .LCPI0_6(%rip), %ymm6   # ymm6 = [-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0]
	VBROADCASTSS(LCPI0_6, Y6)

	//    vaddps  %ymm6, %ymm0, %ymm0
	VADDPS(Y6, Y0, Y0)

	//    vaddps  %ymm5, %ymm0, %ymm0
	VADDPS(Y5, Y0, Y0)

	//    vandps  %ymm3, %ymm4, %ymm3
	VANDPS(Y3, Y4, Y3)

	//    vsubps  %ymm3, %ymm2, %ymm2
	VSUBPS(Y3, Y2, Y2)

	//    vmulps  %ymm0, %ymm0, %ymm3
	VMULPS(Y0, Y0, Y3)

	//    vbroadcastss    .LCPI0_7(%rip), %ymm4   # ymm4 = [7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2]
	VBROADCASTSS(LCPI0_7, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_8(%rip), %ymm5   # ymm5 = [-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1]
	VBROADCASTSS(LCPI0_8, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_9(%rip), %ymm5   # ymm5 = [1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1]
	VBROADCASTSS(LCPI0_9, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_10(%rip), %ymm5  # ymm5 = [-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1]
	VBROADCASTSS(LCPI0_10, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_11(%rip), %ymm5  # ymm5 = [1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1]
	VBROADCASTSS(LCPI0_11, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_12(%rip), %ymm5  # ymm5 = [-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1]
	VBROADCASTSS(LCPI0_12, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_13(%rip), %ymm5  # ymm5 = [2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1]
	VBROADCASTSS(LCPI0_13, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_14(%rip), %ymm5  # ymm5 = [-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1]
	VBROADCASTSS(LCPI0_14, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vbroadcastss    .LCPI0_15(%rip), %ymm5  # ymm5 = [3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1]
	VBROADCASTSS(LCPI0_15, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	VMULPS(Y4, Y0, Y4)

	//    vmulps  %ymm4, %ymm3, %ymm4
	VMULPS(Y4, Y3, Y4)

	//    vbroadcastss    .LCPI0_16(%rip), %ymm5  # ymm5 = [-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4]
	VBROADCASTSS(LCPI0_16, Y5)

	//    vmulps  %ymm5, %ymm2, %ymm5
	VMULPS(Y5, Y2, Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	VADDPS(Y5, Y4, Y4)

	//    vbroadcastss    .LCPI0_17(%rip), %ymm5  # ymm5 = [-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1]
	VBROADCASTSS(LCPI0_17, Y5)

	//    vmulps  %ymm5, %ymm3, %ymm3
	VMULPS(Y5, Y3, Y3)

	//    vaddps  %ymm3, %ymm4, %ymm3
	VADDPS(Y3, Y4, Y3)

	//    vbroadcastss    .LCPI0_18(%rip), %ymm4  # ymm4 = [6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1]
	VBROADCASTSS(LCPI0_18, Y4)

	//    vmulps  %ymm4, %ymm2, %ymm2
	VMULPS(Y4, Y2, Y2)

	//    vaddps  %ymm3, %ymm0, %ymm0
	VADDPS(Y3, Y0, Y0)

	//    vaddps  %ymm0, %ymm2, %ymm0
	VADDPS(Y0, Y2, Y0)

	//    vorps   %ymm0, %ymm1, %ymm0
	VORPS(Y0, Y1, Y0)

	// ---

	VMOVUPS(Y0, y)

	RET()
}

func buildSSE(bits int) {
	globlData4 := func(name string, v U32) Mem {
		m := GLOBL(name, RODATA|NOPTR)
		DATA(0, v)
		DATA(4, v)
		DATA(8, v)
		DATA(12, v)
		return m
	}

	LCPI0_0 := globlData4("SSE_LCPI0_0", 0x00800000)   // float 1.17549435E-38
	LCPI0_1 := globlData4("SSE_LCPI0_1", 2155872255)   // 0x807fffff
	LCPI0_2 := globlData4("SSE_LCPI0_2", 1056964608)   // 0x3f000000
	LCPI0_3 := globlData4("SSE_LCPI0_3", 4294967169)   // 0xffffff81
	LCPI0_4 := globlData4("SSE_LCPI0_4", 0x3f800000)   // float 1
	LCPI0_5 := globlData4("SSE_LCPI0_5", 0x3f3504f3)   // float 0.707106769
	LCPI0_6 := globlData4("SSE_LCPI0_6", 0xbf800000)   // float -1
	LCPI0_7 := globlData4("SSE_LCPI0_7", 0x3d9021bb)   // float 0.0703768358
	LCPI0_8 := globlData4("SSE_LCPI0_8", 0xbdebd1b8)   // float -0.115146101
	LCPI0_9 := globlData4("SSE_LCPI0_9", 0x3def251a)   // float 0.116769984
	LCPI0_10 := globlData4("SSE_LCPI0_10", 0xbdfe5d4f) // float -0.12420141
	LCPI0_11 := globlData4("SSE_LCPI0_11", 0x3e11e9bf) // float 0.142493233
	LCPI0_12 := globlData4("SSE_LCPI0_12", 0xbe2aae50) // float -0.166680574
	LCPI0_13 := globlData4("SSE_LCPI0_13", 0x3e4cceac) // float 0.200007141
	LCPI0_14 := globlData4("SSE_LCPI0_14", 0xbe7ffffc) // float -0.24999994
	LCPI0_15 := globlData4("SSE_LCPI0_15", 0x3eaaaaaa) // float 0.333333313
	LCPI0_16 := globlData4("SSE_LCPI0_16", 0xb95e8083) // float -2.12194442E-4
	LCPI0_17 := globlData4("SSE_LCPI0_17", 0xbf000000) // float -0.5
	LCPI0_18 := globlData4("SSE_LCPI0_18", 0x3f318000) // float 0.693359375

	name := fmt.Sprintf("LogSSE%d", bits)
	signature := fmt.Sprintf("func(x, y []float%d)", bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf(
		"%s computes the natural logarithm of each element of x, storing the result in y (%d bits, SSE required).",
		name, bits,
	))

	x := Mem{Base: Load(Param("x").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}

	MOVUPS(x, X0)

	// ---

	//        xorps   %xmm2, %xmm2
	XORPS(X2, X2)

	//        movaps  %xmm0, %xmm1
	MOVAPS(X0, X1)

	//        cmpleps %xmm2, %xmm1
	CMPPS(X2, X1, U8(2))

	//        maxps   .LCPI0_0(%rip), %xmm0
	MAXPS(LCPI0_0, X0)

	//        movaps  %xmm0, %xmm2
	MOVAPS(X0, X2)

	//        psrld   $23, %xmm2
	PSRLL(U8(23), X2)

	//        andps   .LCPI0_1(%rip), %xmm0
	ANDPS(LCPI0_1, X0)

	//        orps    .LCPI0_2(%rip), %xmm0
	ORPS(LCPI0_2, X0)

	//        paddd   .LCPI0_3(%rip), %xmm2
	PADDD(LCPI0_3, X2)

	//        movaps  %xmm0, %xmm4
	MOVAPS(X0, X4)

	//        cmpltps .LCPI0_5(%rip), %xmm4
	CMPPS(LCPI0_5, X4, U8(1))

	//        movaps  %xmm4, %xmm3
	MOVAPS(X4, X3)

	//        andps   %xmm0, %xmm3
	ANDPS(X0, X3)

	//        addps   .LCPI0_6(%rip), %xmm0
	ADDPS(LCPI0_6, X0)

	//        addps   %xmm3, %xmm0
	ADDPS(X3, X0)

	//        movaps  .LCPI0_7(%rip), %xmm3           # xmm3 = [7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2]
	MOVAPS(LCPI0_7, X3)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_8(%rip), %xmm3
	ADDPS(LCPI0_8, X3)

	//        cvtdq2ps        %xmm2, %xmm2
	CVTPL2PS(X2, X2)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_9(%rip), %xmm3
	ADDPS(LCPI0_9, X3)

	//        movaps  .LCPI0_4(%rip), %xmm5           # xmm5 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	MOVAPS(LCPI0_4, X5)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_10(%rip), %xmm3
	ADDPS(LCPI0_10, X3)

	//        addps   %xmm5, %xmm2
	ADDPS(X5, X2)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_11(%rip), %xmm3
	ADDPS(LCPI0_11, X3)

	//        andps   %xmm5, %xmm4
	ANDPS(X5, X4)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_12(%rip), %xmm3
	ADDPS(LCPI0_12, X3)

	//        subps   %xmm4, %xmm2
	SUBPS(X4, X2)

	//        movaps  %xmm0, %xmm4
	MOVAPS(X0, X4)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_13(%rip), %xmm3
	ADDPS(LCPI0_13, X3)

	//        mulps   %xmm0, %xmm4
	MULPS(X0, X4)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_14(%rip), %xmm3
	ADDPS(LCPI0_14, X3)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        addps   .LCPI0_15(%rip), %xmm3
	ADDPS(LCPI0_15, X3)

	//        mulps   %xmm0, %xmm3
	MULPS(X0, X3)

	//        mulps   %xmm4, %xmm3
	MULPS(X4, X3)

	//        movaps  .LCPI0_16(%rip), %xmm5          # xmm5 = [-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4]
	MOVAPS(LCPI0_16, X5)

	//        mulps   %xmm2, %xmm5
	MULPS(X2, X5)

	//        addps   %xmm3, %xmm5
	ADDPS(X3, X5)

	//        mulps   .LCPI0_17(%rip), %xmm4
	MULPS(LCPI0_17, X4)

	//        mulps   .LCPI0_18(%rip), %xmm2
	MULPS(LCPI0_18, X2)

	//        addps   %xmm5, %xmm4
	ADDPS(X5, X4)

	//        addps   %xmm4, %xmm0
	ADDPS(X4, X0)

	//        addps   %xmm2, %xmm0
	ADDPS(X2, X0)

	//        orps    %xmm1, %xmm0
	ORPS(X1, X0)

	// ---

	MOVUPS(X0, y)

	RET()
}
