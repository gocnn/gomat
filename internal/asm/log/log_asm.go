package main

//go:generate go run . -bits 32 -out ../../../asm/asm32/log_amd64.s -stubs ../../../asm/asm32/log_amd64.go -pkg asm32

import (
	"flag"
	"fmt"

	"github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

func main() {
	var bits = flag.Int("bits", 32, "bits to generate")
	flag.Parse()

	build.ConstraintExpr("amd64,gc,!noasm,!gccgo")

	if *bits == 32 {
		buildAVX(32)
		buildSSE(32)
	}

	build.Generate()
}

func buildAVX(bits int) {
	LCPI0_0 := build.ConstData("AVX2_LCPI0_0", operand.U32(0x00800000))   // float 1.17549435E-38
	LCPI0_1 := build.ConstData("AVX2_LCPI0_1", operand.U32(2155872255))   // 0x807fffff
	LCPI0_2 := build.ConstData("AVX2_LCPI0_2", operand.U32(1056964608))   // 0x3f000000
	LCPI0_3 := build.ConstData("AVX2_LCPI0_3", operand.U32(4294967169))   // 0xffffff81
	LCPI0_4 := build.ConstData("AVX2_LCPI0_4", operand.U32(0x3f800000))   // float 1
	LCPI0_5 := build.ConstData("AVX2_LCPI0_5", operand.U32(0x3f3504f3))   // float 0.707106769
	LCPI0_6 := build.ConstData("AVX2_LCPI0_6", operand.U32(0xbf800000))   // float -1
	LCPI0_7 := build.ConstData("AVX2_LCPI0_7", operand.U32(0x3d9021bb))   // float 0.0703768358
	LCPI0_8 := build.ConstData("AVX2_LCPI0_8", operand.U32(0xbdebd1b8))   // float -0.115146101
	LCPI0_9 := build.ConstData("AVX2_LCPI0_9", operand.U32(0x3def251a))   // float 0.116769984
	LCPI0_10 := build.ConstData("AVX2_LCPI0_10", operand.U32(0xbdfe5d4f)) // float -0.12420141
	LCPI0_11 := build.ConstData("AVX2_LCPI0_11", operand.U32(0x3e11e9bf)) // float 0.142493233
	LCPI0_12 := build.ConstData("AVX2_LCPI0_12", operand.U32(0xbe2aae50)) // float -0.166680574
	LCPI0_13 := build.ConstData("AVX2_LCPI0_13", operand.U32(0x3e4cceac)) // float 0.200007141
	LCPI0_14 := build.ConstData("AVX2_LCPI0_14", operand.U32(0xbe7ffffc)) // float -0.24999994
	LCPI0_15 := build.ConstData("AVX2_LCPI0_15", operand.U32(0x3eaaaaaa)) // float 0.333333313
	LCPI0_16 := build.ConstData("AVX2_LCPI0_16", operand.U32(0xb95e8083)) // float -2.12194442E-4
	LCPI0_17 := build.ConstData("AVX2_LCPI0_17", operand.U32(0xbf000000)) // float -0.5
	LCPI0_18 := build.ConstData("AVX2_LCPI0_18", operand.U32(0x3f318000)) // float 0.693359375

	name := "LogAVX"
	signature := fmt.Sprintf("func(x, y []float%d)", bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf(
		"%s computes the natural logarithm of each element of x, storing the result in y (%d bits, AVX2 required).",
		name, bits,
	))

	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}

	build.VMOVUPS(x, reg.Y0)

	// ---

	//    vxorps  %xmm1, %xmm1, %xmm1
	build.VXORPS(reg.X1, reg.X1, reg.X1)

	//    vcmpleps        %ymm1, %ymm0, %ymm1
	build.VCMPPS(operand.U8(2), reg.Y1, reg.Y0, reg.Y1)

	//    vbroadcastss    .LCPI0_0(%rip), %ymm2   # ymm2 = [1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38,1.17549435E-38]
	build.VBROADCASTSS(LCPI0_0, reg.Y2)

	//    vmaxps  %ymm2, %ymm0, %ymm0
	build.VMAXPS(reg.Y2, reg.Y0, reg.Y0)

	//    vpsrld  $23, %ymm0, %ymm2
	build.VPSRLD(operand.U8(23), reg.Y0, reg.Y2)

	//    vbroadcastss    .LCPI0_1(%rip), %ymm3   # ymm3 = [2155872255,2155872255,2155872255,2155872255,2155872255,2155872255,2155872255,2155872255]
	build.VBROADCASTSS(LCPI0_1, reg.Y3)

	//    vandps  %ymm3, %ymm0, %ymm0
	build.VANDPS(reg.Y3, reg.Y0, reg.Y0)

	//    vbroadcastss    .LCPI0_2(%rip), %ymm3   # ymm3 = [1056964608,1056964608,1056964608,1056964608,1056964608,1056964608,1056964608,1056964608]
	build.VBROADCASTSS(LCPI0_2, reg.Y3)

	//    vpbroadcastd    .LCPI0_3(%rip), %ymm4   # ymm4 = [4294967169,4294967169,4294967169,4294967169,4294967169,4294967169,4294967169,4294967169]
	build.VPBROADCASTD(LCPI0_3, reg.Y4)

	//    vorps   %ymm3, %ymm0, %ymm0
	build.VORPS(reg.Y3, reg.Y0, reg.Y0)

	//    vpaddd  %ymm4, %ymm2, %ymm2
	build.VPADDD(reg.Y4, reg.Y2, reg.Y2)

	//    vcvtdq2ps       %ymm2, %ymm2
	build.VCVTDQ2PS(reg.Y2, reg.Y2)

	//    vbroadcastss    .LCPI0_4(%rip), %ymm3   # ymm3 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	build.VBROADCASTSS(LCPI0_4, reg.Y3)

	//    vaddps  %ymm3, %ymm2, %ymm2
	build.VADDPS(reg.Y3, reg.Y2, reg.Y2)

	//    vbroadcastss    .LCPI0_5(%rip), %ymm4   # ymm4 = [7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1,7.07106769E-1]
	build.VBROADCASTSS(LCPI0_5, reg.Y4)

	//    vcmpltps        %ymm4, %ymm0, %ymm4
	build.VCMPPS(operand.U8(1), reg.Y4, reg.Y0, reg.Y4)

	//    vandps  %ymm0, %ymm4, %ymm5
	build.VANDPS(reg.Y0, reg.Y4, reg.Y5)

	//    vbroadcastss    .LCPI0_6(%rip), %ymm6   # ymm6 = [-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0,-1.0E+0]
	build.VBROADCASTSS(LCPI0_6, reg.Y6)

	//    vaddps  %ymm6, %ymm0, %ymm0
	build.VADDPS(reg.Y6, reg.Y0, reg.Y0)

	//    vaddps  %ymm5, %ymm0, %ymm0
	build.VADDPS(reg.Y5, reg.Y0, reg.Y0)

	//    vandps  %ymm3, %ymm4, %ymm3
	build.VANDPS(reg.Y3, reg.Y4, reg.Y3)

	//    vsubps  %ymm3, %ymm2, %ymm2
	build.VSUBPS(reg.Y3, reg.Y2, reg.Y2)

	//    vmulps  %ymm0, %ymm0, %ymm3
	build.VMULPS(reg.Y0, reg.Y0, reg.Y3)

	//    vbroadcastss    .LCPI0_7(%rip), %ymm4   # ymm4 = [7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2]
	build.VBROADCASTSS(LCPI0_7, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_8(%rip), %ymm5   # ymm5 = [-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1,-1.15146101E-1]
	build.VBROADCASTSS(LCPI0_8, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_9(%rip), %ymm5   # ymm5 = [1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1,1.16769984E-1]
	build.VBROADCASTSS(LCPI0_9, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_10(%rip), %ymm5  # ymm5 = [-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1,-1.2420141E-1]
	build.VBROADCASTSS(LCPI0_10, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_11(%rip), %ymm5  # ymm5 = [1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1,1.42493233E-1]
	build.VBROADCASTSS(LCPI0_11, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_12(%rip), %ymm5  # ymm5 = [-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1,-1.66680574E-1]
	build.VBROADCASTSS(LCPI0_12, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_13(%rip), %ymm5  # ymm5 = [2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1,2.00007141E-1]
	build.VBROADCASTSS(LCPI0_13, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_14(%rip), %ymm5  # ymm5 = [-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1,-2.4999994E-1]
	build.VBROADCASTSS(LCPI0_14, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vbroadcastss    .LCPI0_15(%rip), %ymm5  # ymm5 = [3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1,3.33333313E-1]
	build.VBROADCASTSS(LCPI0_15, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vmulps  %ymm4, %ymm0, %ymm4
	build.VMULPS(reg.Y4, reg.Y0, reg.Y4)

	//    vmulps  %ymm4, %ymm3, %ymm4
	build.VMULPS(reg.Y4, reg.Y3, reg.Y4)

	//    vbroadcastss    .LCPI0_16(%rip), %ymm5  # ymm5 = [-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4]
	build.VBROADCASTSS(LCPI0_16, reg.Y5)

	//    vmulps  %ymm5, %ymm2, %ymm5
	build.VMULPS(reg.Y5, reg.Y2, reg.Y5)

	//    vaddps  %ymm5, %ymm4, %ymm4
	build.VADDPS(reg.Y5, reg.Y4, reg.Y4)

	//    vbroadcastss    .LCPI0_17(%rip), %ymm5  # ymm5 = [-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1,-5.0E-1]
	build.VBROADCASTSS(LCPI0_17, reg.Y5)

	//    vmulps  %ymm5, %ymm3, %ymm3
	build.VMULPS(reg.Y5, reg.Y3, reg.Y3)

	//    vaddps  %ymm3, %ymm4, %ymm3
	build.VADDPS(reg.Y3, reg.Y4, reg.Y3)

	//    vbroadcastss    .LCPI0_18(%rip), %ymm4  # ymm4 = [6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1,6.93359375E-1]
	build.VBROADCASTSS(LCPI0_18, reg.Y4)

	//    vmulps  %ymm4, %ymm2, %ymm2
	build.VMULPS(reg.Y4, reg.Y2, reg.Y2)

	//    vaddps  %ymm3, %ymm0, %ymm0
	build.VADDPS(reg.Y3, reg.Y0, reg.Y0)

	//    vaddps  %ymm0, %ymm2, %ymm0
	build.VADDPS(reg.Y0, reg.Y2, reg.Y0)

	//    vorps   %ymm0, %ymm1, %ymm0
	build.VORPS(reg.Y0, reg.Y1, reg.Y0)

	// ---

	build.VMOVUPS(reg.Y0, y)

	build.RET()
}

func buildSSE(bits int) {
	globlData4 := func(name string, v operand.U32) operand.Mem {
		m := build.GLOBL(name, build.RODATA|build.NOPTR)
		build.DATA(0, v)
		build.DATA(4, v)
		build.DATA(8, v)
		build.DATA(12, v)
		return m
	}

	LCPI0_0 := globlData4("SSE_LCPI0_0", operand.U32(0x00800000))   // float 1.17549435E-38
	LCPI0_1 := globlData4("SSE_LCPI0_1", operand.U32(2155872255))   // 0x807fffff
	LCPI0_2 := globlData4("SSE_LCPI0_2", operand.U32(1056964608))   // 0x3f000000
	LCPI0_3 := globlData4("SSE_LCPI0_3", operand.U32(4294967169))   // 0xffffff81
	LCPI0_4 := globlData4("SSE_LCPI0_4", operand.U32(0x3f800000))   // float 1
	LCPI0_5 := globlData4("SSE_LCPI0_5", operand.U32(0x3f3504f3))   // float 0.707106769
	LCPI0_6 := globlData4("SSE_LCPI0_6", operand.U32(0xbf800000))   // float -1
	LCPI0_7 := globlData4("SSE_LCPI0_7", operand.U32(0x3d9021bb))   // float 0.0703768358
	LCPI0_8 := globlData4("SSE_LCPI0_8", operand.U32(0xbdebd1b8))   // float -0.115146101
	LCPI0_9 := globlData4("SSE_LCPI0_9", operand.U32(0x3def251a))   // float 0.116769984
	LCPI0_10 := globlData4("SSE_LCPI0_10", operand.U32(0xbdfe5d4f)) // float -0.12420141
	LCPI0_11 := globlData4("SSE_LCPI0_11", operand.U32(0x3e11e9bf)) // float 0.142493233
	LCPI0_12 := globlData4("SSE_LCPI0_12", operand.U32(0xbe2aae50)) // float -0.166680574
	LCPI0_13 := globlData4("SSE_LCPI0_13", operand.U32(0x3e4cceac)) // float 0.200007141
	LCPI0_14 := globlData4("SSE_LCPI0_14", operand.U32(0xbe7ffffc)) // float -0.24999994
	LCPI0_15 := globlData4("SSE_LCPI0_15", operand.U32(0x3eaaaaaa)) // float 0.333333313
	LCPI0_16 := globlData4("SSE_LCPI0_16", operand.U32(0xb95e8083)) // float -2.12194442E-4
	LCPI0_17 := globlData4("SSE_LCPI0_17", operand.U32(0xbf000000)) // float -0.5
	LCPI0_18 := globlData4("SSE_LCPI0_18", operand.U32(0x3f318000)) // float 0.693359375

	name := "LogSSE"
	signature := fmt.Sprintf("func(x, y []float%d)", bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf(
		"%s computes the natural logarithm of each element of x, storing the result in y (%d bits, SSE required).",
		name, bits,
	))

	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}

	build.MOVUPS(x, reg.X0)

	// ---

	//        xorps   %xmm2, %xmm2
	build.XORPS(reg.X2, reg.X2)

	//        movaps  %xmm0, %xmm1
	build.MOVAPS(reg.X0, reg.X1)

	//        cmpleps %xmm2, %xmm1
	build.CMPPS(reg.X2, reg.X1, operand.U8(2))

	//        maxps   .LCPI0_0(%rip), %xmm0
	build.MAXPS(LCPI0_0, reg.X0)

	//        movaps  %xmm0, %xmm2
	build.MOVAPS(reg.X0, reg.X2)

	//        psrld   $23, %xmm2
	build.PSRLL(operand.U8(23), reg.X2)

	//        andps   .LCPI0_1(%rip), %xmm0
	build.ANDPS(LCPI0_1, reg.X0)

	//        orps    .LCPI0_2(%rip), %xmm0
	build.ORPS(LCPI0_2, reg.X0)

	//        paddd   .LCPI0_3(%rip), %xmm2
	build.PADDD(LCPI0_3, reg.X2)

	//        movaps  %xmm0, %xmm4
	build.MOVAPS(reg.X0, reg.X4)

	//        cmpltps .LCPI0_5(%rip), %xmm4
	build.CMPPS(LCPI0_5, reg.X4, operand.U8(1))

	//        movaps  %xmm4, %xmm3
	build.MOVAPS(reg.X4, reg.X3)

	//        andps   %xmm0, %xmm3
	build.ANDPS(reg.X0, reg.X3)

	//        addps   .LCPI0_6(%rip), %xmm0
	build.ADDPS(LCPI0_6, reg.X0)

	//        addps   %xmm3, %xmm0
	build.ADDPS(reg.X3, reg.X0)

	//        movaps  .LCPI0_7(%rip), %xmm3           # xmm3 = [7.03768358E-2,7.03768358E-2,7.03768358E-2,7.03768358E-2]
	build.MOVAPS(LCPI0_7, reg.X3)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_8(%rip), %xmm3
	build.ADDPS(LCPI0_8, reg.X3)

	//        cvtdq2ps        %xmm2, %xmm2
	build.CVTPL2PS(reg.X2, reg.X2)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_9(%rip), %xmm3
	build.ADDPS(LCPI0_9, reg.X3)

	//        movaps  .LCPI0_4(%rip), %xmm5           # xmm5 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	build.MOVAPS(LCPI0_4, reg.X5)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_10(%rip), %xmm3
	build.ADDPS(LCPI0_10, reg.X3)

	//        addps   %xmm5, %xmm2
	build.ADDPS(reg.X5, reg.X2)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_11(%rip), %xmm3
	build.ADDPS(LCPI0_11, reg.X3)

	//        andps   %xmm5, %xmm4
	build.ANDPS(reg.X5, reg.X4)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_12(%rip), %xmm3
	build.ADDPS(LCPI0_12, reg.X3)

	//        subps   %xmm4, %xmm2
	build.SUBPS(reg.X4, reg.X2)

	//        movaps  %xmm0, %xmm4
	build.MOVAPS(reg.X0, reg.X4)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_13(%rip), %xmm3
	build.ADDPS(LCPI0_13, reg.X3)

	//        mulps   %xmm0, %xmm4
	build.MULPS(reg.X0, reg.X4)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_14(%rip), %xmm3
	build.ADDPS(LCPI0_14, reg.X3)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        addps   .LCPI0_15(%rip), %xmm3
	build.ADDPS(LCPI0_15, reg.X3)

	//        mulps   %xmm0, %xmm3
	build.MULPS(reg.X0, reg.X3)

	//        mulps   %xmm4, %xmm3
	build.MULPS(reg.X4, reg.X3)

	//        movaps  .LCPI0_16(%rip), %xmm5          # xmm5 = [-2.12194442E-4,-2.12194442E-4,-2.12194442E-4,-2.12194442E-4]
	build.MOVAPS(LCPI0_16, reg.X5)

	//        mulps   %xmm2, %xmm5
	build.MULPS(reg.X2, reg.X5)

	//        addps   %xmm3, %xmm5
	build.ADDPS(reg.X3, reg.X5)

	//        mulps   .LCPI0_17(%rip), %xmm4
	build.MULPS(LCPI0_17, reg.X4)

	//        mulps   .LCPI0_18(%rip), %xmm2
	build.MULPS(LCPI0_18, reg.X2)

	//        addps   %xmm5, %xmm4
	build.ADDPS(reg.X5, reg.X4)

	//        addps   %xmm4, %xmm0
	build.ADDPS(reg.X4, reg.X0)

	//        addps   %xmm2, %xmm0
	build.ADDPS(reg.X2, reg.X0)

	//        orps    %xmm1, %xmm0
	build.ORPS(reg.X1, reg.X0)

	// ---

	build.MOVUPS(reg.X0, y)

	build.RET()
}
