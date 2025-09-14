package main

//go:generate go run . -bits 32 -out ../../mat32/exp_amd64.s -stubs ../../mat32/exp_amd64.go -pkg mat32

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
		buildAVX32()
		buildSSE32()
	}

	build.Generate()
}

func buildAVX32() {
	expHi := build.ConstData("expHi", operand.U32(0x42b0c0a5))               // float 88.3762589
	expLo := build.ConstData("expLo", operand.U32(0xc2b0c0a5))               // float -88.3762589
	cephesLOG2EF := build.ConstData("cephesLOG2EF", operand.U32(0x3fb8aa3b)) // float 1.44269502
	zp5 := build.ConstData("zp5", operand.U32(0x3f000000))                   // float 0.5
	one := build.ConstData("one", operand.U32(0x3f800000))                   // float 1
	cephesExpC1 := build.ConstData("cephesExpC1", operand.U32(0xbf318000))   // float -0.693359375
	cephesExpC2 := build.ConstData("cephesExpC2", operand.U32(0x395e8083))   // float 2.12194442E-4
	cephesExpP0 := build.ConstData("cephesExpP0", operand.U32(0x39506967))   // float 1.98756912E-4
	cephesExpP1 := build.ConstData("cephesExpP1", operand.U32(0x3ab743ce))   //float 0.00139819994
	cephesExpP2 := build.ConstData("cephesExpP2", operand.U32(0x3c088908))   // float 0.00833345205
	cephesExpP3 := build.ConstData("cephesExpP3", operand.U32(0x3d2aa9c1))   // float 0.0416657962
	cephesExpP4 := build.ConstData("cephesExpP4", operand.U32(0x3e2aaaaa))   // float 0.166666657

	const name = "ExpAVX"
	const signature = "func(x, y []float32)"
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf(
		"%s computes the base-e exponential of each element of x, storing the result in y (32 bits, AVX required).",
		name,
	))

	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}

	build.VMOVUPS(x, reg.Y0)

	// ---

	//  x = _mm256_min_ps(x, *(v8sf*)_ps256_expHi);
	//        vbroadcastss    .expHi(%rip), %ymm1   # ymm1 = [8.83762589E+1,8.83762589E+1,8.83762589E+1,8.83762589E+1,8.83762589E+1,8.83762589E+1,8.83762589E+1,8.83762589E+1]
	//        vminps  %ymm1, %ymm0, %ymm0
	build.VBROADCASTSS(expHi, reg.Y1)
	build.VMINPS(reg.Y1, reg.Y0, reg.Y0)

	//  x = _mm256_max_ps(x, *(v8sf*)_ps256_expLo);
	//        vbroadcastss    .expLo(%rip), %ymm1   # ymm1 = [-8.83762589E+1,-8.83762589E+1,-8.83762589E+1,-8.83762589E+1,-8.83762589E+1,-8.83762589E+1,-8.83762589E+1,-8.83762589E+1]
	//        vmaxps  %ymm1, %ymm0, %ymm0
	build.VBROADCASTSS(expLo, reg.Y1)
	build.VMAXPS(reg.Y1, reg.Y0, reg.Y0)

	//  /* express exp(x) as exp(g + n*log(2)) */
	//  fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
	//        vbroadcastss    .cephesLOG2EF(%rip), %ymm1   # ymm1 = [1.44269502E+0,1.44269502E+0,1.44269502E+0,1.44269502E+0,1.44269502E+0,1.44269502E+0,1.44269502E+0,1.44269502E+0]
	//        vmulps  %ymm1, %ymm0, %ymm1
	build.VBROADCASTSS(cephesLOG2EF, reg.Y1)
	build.VMULPS(reg.Y1, reg.Y0, reg.Y1)

	//  fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);
	//        vbroadcastss    .zp5(%rip), %ymm2   # ymm2 = [5.0E-1,5.0E-1,5.0E-1,5.0E-1,5.0E-1,5.0E-1,5.0E-1,5.0E-1]
	//        vaddps  %ymm2, %ymm1, %ymm1
	build.VBROADCASTSS(zp5, reg.Y2)
	build.VADDPS(reg.Y2, reg.Y1, reg.Y1)

	//  /* how to perform a floorf with SSE: just below */
	//  //imm0 = _mm256_cvttps_epi32(fx);
	//  //tmp  = _mm256_cvtepi32_ps(imm0);
	//
	//  tmp = _mm256_floor_ps(fx);
	//        vroundps        $1, %ymm1, %ymm3
	build.VROUNDPS(operand.U8(1), reg.Y1, reg.Y3)

	//  /* if greater, substract 1 */
	//  //v8sf mask = _mm256_cmpgt_ps(tmp, fx);
	//  v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
	//        vcmpltps        %ymm3, %ymm1, %ymm1
	build.VCMPPS(operand.U8(1), reg.Y3, reg.Y1, reg.Y1)

	//   mask = _mm256_and_ps(mask, one);
	//        vbroadcastss    .one(%rip), %ymm4   # ymm4 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	//        vandps  %ymm4, %ymm1, %ymm1
	build.VBROADCASTSS(one, reg.Y4)
	build.VANDPS(reg.Y4, reg.Y1, reg.Y1)

	//  fx = _mm256_sub_ps(tmp, mask);
	//        vsubps  %ymm1, %ymm3, %ymm1
	build.VSUBPS(reg.Y1, reg.Y3, reg.Y1)

	//  tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
	//        vbroadcastss    .cephesExpC1(%rip), %ymm3   # ymm3 = [-6.93359375E-1,-6.93359375E-1,-6.93359375E-1,-6.93359375E-1,-6.93359375E-1,-6.93359375E-1,-6.93359375E-1,-6.93359375E-1]
	//        vmulps  %ymm3, %ymm1, %ymm3
	build.VBROADCASTSS(cephesExpC1, reg.Y3)
	build.VMULPS(reg.Y3, reg.Y1, reg.Y3)

	//  v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
	//        vbroadcastss    .cephesExpC2(%rip), %ymm5   # ymm5 = [2.12194442E-4,2.12194442E-4,2.12194442E-4,2.12194442E-4,2.12194442E-4,2.12194442E-4,2.12194442E-4,2.12194442E-4]
	//        vmulps  %ymm5, %ymm1, %ymm5
	build.VBROADCASTSS(cephesExpC2, reg.Y5)
	build.VMULPS(reg.Y5, reg.Y1, reg.Y5)

	//  x = _mm256_sub_ps(x, tmp);
	//        vaddps  %ymm3, %ymm0, %ymm0
	build.VADDPS(reg.Y3, reg.Y0, reg.Y0)

	//  x = _mm256_sub_ps(x, z);
	//        vaddps  %ymm5, %ymm0, %ymm0
	build.VADDPS(reg.Y5, reg.Y0, reg.Y0)

	//  z = _mm256_mul_ps(x,x);
	//        vmulps  %ymm0, %ymm0, %ymm3
	build.VMULPS(reg.Y0, reg.Y0, reg.Y3)

	//  v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
	//  y = _mm256_mul_ps(y, x);
	//        vbroadcastss    .cephesExpP0(%rip), %ymm5   # ymm5 = [1.98756912E-4,1.98756912E-4,1.98756912E-4,1.98756912E-4,1.98756912E-4,1.98756912E-4,1.98756912E-4,1.98756912E-4]
	//        vmulps  %ymm5, %ymm0, %ymm5
	build.VBROADCASTSS(cephesExpP0, reg.Y5)
	build.VMULPS(reg.Y5, reg.Y0, reg.Y5)

	//  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
	//        vbroadcastss    .cephesExpP1(%rip), %ymm6   # ymm6 = [1.39819994E-3,1.39819994E-3,1.39819994E-3,1.39819994E-3,1.39819994E-3,1.39819994E-3,1.39819994E-3,1.39819994E-3]
	//        vaddps  %ymm6, %ymm5, %ymm5
	build.VBROADCASTSS(cephesExpP1, reg.Y6)
	build.VADDPS(reg.Y6, reg.Y5, reg.Y5)

	//  y = _mm256_mul_ps(y, x);
	//        vmulps  %ymm5, %ymm0, %ymm5
	build.VMULPS(reg.Y5, reg.Y0, reg.Y5)

	//  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
	//        vbroadcastss    .cephesExpP2(%rip), %ymm6   # ymm6 = [8.33345205E-3,8.33345205E-3,8.33345205E-3,8.33345205E-3,8.33345205E-3,8.33345205E-3,8.33345205E-3,8.33345205E-3]
	//        vaddps  %ymm6, %ymm5, %ymm5
	build.VBROADCASTSS(cephesExpP2, reg.Y6)
	build.VADDPS(reg.Y6, reg.Y5, reg.Y5)

	//  y = _mm256_mul_ps(y, x);
	//        vmulps  %ymm5, %ymm0, %ymm5
	build.VMULPS(reg.Y5, reg.Y0, reg.Y5)

	//  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
	//        vbroadcastss    .cephesExpP3(%rip), %ymm6  # ymm6 = [4.16657962E-2,4.16657962E-2,4.16657962E-2,4.16657962E-2,4.16657962E-2,4.16657962E-2,4.16657962E-2,4.16657962E-2]
	//        vaddps  %ymm6, %ymm5, %ymm5
	build.VBROADCASTSS(cephesExpP3, reg.Y6)
	build.VADDPS(reg.Y6, reg.Y5, reg.Y5)

	//  y = _mm256_mul_ps(y, x);
	//        vmulps  %ymm5, %ymm0, %ymm5
	build.VMULPS(reg.Y5, reg.Y0, reg.Y5)

	//  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
	//        vbroadcastss    .cephesExpP4(%rip), %ymm6  # ymm6 = [1.66666657E-1,1.66666657E-1,1.66666657E-1,1.66666657E-1,1.66666657E-1,1.66666657E-1,1.66666657E-1,1.66666657E-1]
	//        vaddps  %ymm6, %ymm5, %ymm5
	build.VBROADCASTSS(cephesExpP4, reg.Y6)
	build.VADDPS(reg.Y6, reg.Y5, reg.Y5)

	//  y = _mm256_mul_ps(y, x);
	//        vmulps  %ymm5, %ymm0, %ymm5
	build.VMULPS(reg.Y5, reg.Y0, reg.Y5)

	//  y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
	//        vaddps  %ymm2, %ymm5, %ymm2
	build.VADDPS(reg.Y2, reg.Y5, reg.Y2)

	//  y = _mm256_mul_ps(y, z);
	//        vmulps  %ymm2, %ymm3, %ymm2
	build.VMULPS(reg.Y2, reg.Y3, reg.Y2)

	//  y = _mm256_add_ps(y, x);
	//        vaddps  %ymm2, %ymm0, %ymm0
	build.VADDPS(reg.Y2, reg.Y0, reg.Y0)

	//  y = _mm256_add_ps(y, one);
	//        vaddps  %ymm4, %ymm0, %ymm0
	build.VADDPS(reg.Y4, reg.Y0, reg.Y0)

	//  /* build 2^n */
	//  imm0 = _mm256_cvttps_epi32(fx);
	//        vcvttps2dq      %ymm1, %ymm1
	build.VCVTTPS2DQ(reg.Y1, reg.Y1)

	//  imm0 = avx2_mm256_slli_epi32(imm0, 23);
	//        vpslld  $23, %ymm1, %ymm1
	//        vpbroadcastd    .one(%rip), %ymm2   # ymm2 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	//        vpaddd  %ymm2, %ymm1, %ymm1
	build.VPSLLD(operand.U8(23), reg.Y1, reg.Y1)
	build.VPBROADCASTD(one, reg.Y2)
	build.VPADDD(reg.Y2, reg.Y1, reg.Y1)

	//  y = _mm256_mul_ps(y, pow2n);
	//        vmulps  %ymm1, %ymm0, %ymm0
	build.VMULPS(reg.Y1, reg.Y0, reg.Y0)

	// ---

	build.VMOVUPS(reg.Y0, y)

	build.RET()
}

func buildSSE32() {
	globlData4 := func(name string, v operand.U32) operand.Mem {
		m := build.GLOBL(name, build.RODATA|build.NOPTR)
		build.DATA(0, v)
		build.DATA(4, v)
		build.DATA(8, v)
		build.DATA(12, v)
		return m
	}

	LCPI0_0 := globlData4("SSE_LCPI0_0", operand.U32(0x42b0c0a5))   // float 88.3762589
	LCPI0_1 := globlData4("SSE_LCPI0_1", operand.U32(0xc2b0c0a5))   // float -88.3762589
	LCPI0_2 := globlData4("SSE_LCPI0_2", operand.U32(0x3fb8aa3b))   // float 1.44269502
	LCPI0_3 := globlData4("SSE_LCPI0_3", operand.U32(0x3f000000))   // float 0.5
	LCPI0_4 := globlData4("SSE_LCPI0_4", operand.U32(0x3f800000))   // float 1
	LCPI0_5 := globlData4("SSE_LCPI0_5", operand.U32(0xbf318000))   // float -0.693359375
	LCPI0_6 := globlData4("SSE_LCPI0_6", operand.U32(0x395e8083))   // float 2.12194442E-4
	LCPI0_7 := globlData4("SSE_LCPI0_7", operand.U32(0x39506967))   // float 1.98756912E-4
	LCPI0_8 := globlData4("SSE_LCPI0_8", operand.U32(0x3ab743ce))   // float 0.00139819994
	LCPI0_9 := globlData4("SSE_LCPI0_9", operand.U32(0x3c088908))   // float 0.00833345205
	LCPI0_10 := globlData4("SSE_LCPI0_10", operand.U32(0x3d2aa9c1)) // float 0.0416657962
	LCPI0_11 := globlData4("SSE_LCPI0_11", operand.U32(0x3e2aaaaa)) // float 0.166666657

	const name = "ExpSSE"
	const signature = "func(x, y []float32)"
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf(
		"%s computes the base-e exponential of each element of x, storing the result in y (32 bits, SSE required).",
		name,
	))

	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}

	build.MOVUPS(x, reg.X0)

	// ---

	//  v4sf tmp = _mm_setzero_ps(), fx;
	//  v4si emm0;
	//  v4sf one = *(v4sf*)_ps_1;
	//
	//  x = _mm_min_ps(x, *(v4sf*)_ps_exp_hi);
	//        minps   .LCPI0_0(%rip), %xmm0
	build.MINPS(LCPI0_0, reg.X0)

	//  x = _mm_max_ps(x, *(v4sf*)_ps_exp_lo);
	//        maxps   .LCPI0_1(%rip), %xmm0
	build.MAXPS(LCPI0_1, reg.X0)

	//        movaps  .LCPI0_2(%rip), %xmm4           # xmm4 = [1.44269502E+0,1.44269502E+0,1.44269502E+0,1.44269502E+0]
	build.MOVAPS(LCPI0_2, reg.X4)

	//  /* express exp(x) as exp(g + n*log(2)) */
	//  fx = _mm_mul_ps(x, *(v4sf*)_ps_cephes_LOG2EF);
	//        mulps   %xmm0, %xmm4
	build.MULPS(reg.X0, reg.X4)

	//  fx = _mm_add_ps(fx, *(v4sf*)_ps_0p5);
	//        movaps  .LCPI0_3(%rip), %xmm2           # xmm2 = [5.0E-1,5.0E-1,5.0E-1,5.0E-1]
	//        addps   %xmm2, %xmm4
	build.MOVAPS(LCPI0_3, reg.X2)
	build.ADDPS(reg.X2, reg.X4)

	//  /* how to perform a floorf with SSE: just below */
	//
	//  emm0 = _mm_cvttps_epi32(fx);
	//        cvttps2dq       %xmm4, %xmm1
	build.CVTTPS2PL(reg.X4, reg.X1)

	//  tmp  = _mm_cvtepi32_ps(emm0);
	//        cvtdq2ps        %xmm1, %xmm1
	build.CVTPL2PS(reg.X1, reg.X1)

	//  /* if greater, substract 1 */
	//  v4sf mask = _mm_cmpgt_ps(tmp, fx);
	//        cmpltps %xmm1, %xmm4
	build.CMPPS(reg.X1, reg.X4, operand.U8(1))

	//  mask = _mm_and_ps(mask, one);
	//        movaps  .LCPI0_4(%rip), %xmm3           # xmm3 = [1.0E+0,1.0E+0,1.0E+0,1.0E+0]
	//        andps   %xmm3, %xmm4
	build.MOVAPS(LCPI0_4, reg.X3)
	build.ANDPS(reg.X3, reg.X4)

	//  fx = _mm_sub_ps(tmp, mask);
	//        subps   %xmm4, %xmm1
	build.SUBPS(reg.X4, reg.X1)

	//        movaps  .LCPI0_5(%rip), %xmm4           # xmm4 = [-6.93359375E-1,-6.93359375E-1,-6.93359375E-1,-6.93359375E-1]
	build.MOVAPS(LCPI0_5, reg.X4)

	//  tmp = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C1);
	//        mulps   %xmm1, %xmm4
	//        movaps  .LCPI0_6(%rip), %xmm5           # xmm5 = [2.12194442E-4,2.12194442E-4,2.12194442E-4,2.12194442E-4]
	build.MULPS(reg.X1, reg.X4)
	build.MOVAPS(LCPI0_6, reg.X5)

	//  v4sf z = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C2);
	//        mulps   %xmm1, %xmm5
	build.MULPS(reg.X1, reg.X5)
	//  x = _mm_sub_ps(x, tmp);
	//        addps   %xmm4, %xmm0
	build.ADDPS(reg.X4, reg.X0)

	//  x = _mm_sub_ps(x, z);
	//        addps   %xmm5, %xmm0
	build.ADDPS(reg.X5, reg.X0)

	//  z = _mm_mul_ps(x,x);
	//        movaps  %xmm0, %xmm4
	//        movaps  .LCPI0_7(%rip), %xmm5           # xmm5 = [1.98756912E-4,1.98756912E-4,1.98756912E-4,1.98756912E-4]
	build.MOVAPS(reg.X0, reg.X4)
	build.MOVAPS(LCPI0_7, reg.X5)

	//  v4sf y = *(v4sf*)_ps_cephes_exp_p0;
	//  y = _mm_mul_ps(y, x);
	//        mulps   %xmm0, %xmm5
	build.MULPS(reg.X0, reg.X5)

	//  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p1);
	//        addps   .LCPI0_8(%rip), %xmm5
	build.ADDPS(LCPI0_8, reg.X5)

	// ^^ ^^
	//        mulps   %xmm0, %xmm4
	build.MULPS(reg.X0, reg.X4)

	//  y = _mm_mul_ps(y, x);
	//        mulps   %xmm0, %xmm5
	build.MULPS(reg.X0, reg.X5)

	//y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p2);
	//        addps   .LCPI0_9(%rip), %xmm5
	build.ADDPS(LCPI0_9, reg.X5)

	//  y = _mm_mul_ps(y, x);
	//        mulps   %xmm0, %xmm5
	build.MULPS(reg.X0, reg.X5)

	//  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p3);
	//        addps   .LCPI0_10(%rip), %xmm5
	build.ADDPS(LCPI0_10, reg.X5)

	//  y = _mm_mul_ps(y, x);
	//        mulps   %xmm0, %xmm5
	build.MULPS(reg.X0, reg.X5)

	//  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p4);
	//        addps   .LCPI0_11(%rip), %xmm5
	build.ADDPS(LCPI0_11, reg.X5)

	//  y = _mm_mul_ps(y, x);
	//        mulps   %xmm0, %xmm5
	build.MULPS(reg.X0, reg.X5)

	//  y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p5);
	//        addps   %xmm2, %xmm5
	build.ADDPS(reg.X2, reg.X5)

	//  y = _mm_mul_ps(y, z);
	//        mulps   %xmm4, %xmm5
	build.MULPS(reg.X4, reg.X5)

	//  y = _mm_add_ps(y, x);
	//        addps   %xmm5, %xmm0
	build.ADDPS(reg.X5, reg.X0)

	//  y = _mm_add_ps(y, one);
	//        addps   %xmm3, %xmm0
	build.ADDPS(reg.X3, reg.X0)

	//  /* build 2^n */
	//
	//  emm0 = _mm_cvttps_epi32(fx);
	//        cvttps2dq       %xmm1, %xmm1
	build.CVTTPS2PL(reg.X1, reg.X1)

	//  emm0 = _mm_add_epi32(emm0, *(v4si*)_pi32_0x7f);
	//  emm0 = _mm_slli_epi32(emm0, 23);
	//        pslld   $23, %xmm1
	//        paddd   .LCPI0_4(%rip), %xmm1
	build.PSLLL(operand.U8(23), reg.X1)
	build.PADDD(LCPI0_4, reg.X1)

	//  v4sf pow2n = _mm_castsi128_ps(emm0);
	//
	//  y = _mm_mul_ps(y, pow2n);
	//        mulps   %xmm1, %xmm0
	build.MULPS(reg.X1, reg.X0)

	//  return y;
	//        retq

	// ---

	build.MOVUPS(reg.X0, y)

	build.RET()
}
