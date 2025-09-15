package main

//go:generate go run . -bits 64 -out ../../../asm/asm64/dotprod_amd64.s -stubs ../../../asm/asm64/dotprod_amd64.go -pkg asm64
//go:generate go run . -bits 32 -out ../../../asm/asm32/dotprod_amd64.s -stubs ../../../asm/asm32/dotprod_amd64.go -pkg asm32

import (
	"flag"
	"fmt"

	"github.com/mmcloughlin/avo/build"
	"github.com/mmcloughlin/avo/operand"
	"github.com/mmcloughlin/avo/reg"
)

func main() {
	var bits = flag.Int("bits", 64, "bits to generate")
	flag.Parse()

	build.ConstraintExpr("amd64,gc,!noasm,!gccgo")

	if *bits == 32 {
		buildAVX(32)
		buildSSE(32)
	} else {
		buildAVX(64)
		buildSSE(64)
	}

	build.Generate()
}

type bitsToFuncOps map[int]func(...operand.Op)
type bitsToFunc2Ops map[int]func(operand.Op, operand.Op)

var (
	ADDP       = bitsToFunc2Ops{32: build.ADDPS, 64: build.ADDPD}
	ADDS       = bitsToFunc2Ops{32: build.ADDSS, 64: build.ADDSD}
	HADDP      = bitsToFunc2Ops{32: build.HADDPS, 64: build.HADDPD}
	MOVS       = bitsToFunc2Ops{32: build.MOVSS, 64: build.MOVSD}
	MOVUP      = bitsToFunc2Ops{32: build.MOVUPS, 64: build.MOVUPD}
	MULP       = bitsToFunc2Ops{32: build.MULPS, 64: build.MULPD}
	MULS       = bitsToFunc2Ops{32: build.MULSS, 64: build.MULSD}
	VADDP      = bitsToFuncOps{32: build.VADDPS, 64: build.VADDPD}
	VFMADD231P = bitsToFuncOps{32: build.VFMADD231PS, 64: build.VFMADD231PD}
	VFMADD231S = bitsToFuncOps{32: build.VFMADD231SS, 64: build.VFMADD231SD}
	VMOVS      = bitsToFuncOps{32: build.VMOVSS, 64: build.VMOVSD}
	VMOVUP     = bitsToFuncOps{32: build.VMOVUPS, 64: build.VMOVUPD}
	VXORP      = bitsToFuncOps{32: build.VXORPS, 64: build.VXORPD}
	XORP       = bitsToFunc2Ops{32: build.XORPS, 64: build.XORPD}
)

func buildAVX(bits int) {
	name := "DotProdAVX"
	signature := fmt.Sprintf("func(x1, x2 []float%d) float%d", bits, bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s returns the dot product between x1 and x2 (%d bits, AVX required).", name, bits))

	x1 := operand.Mem{Base: build.Load(build.Param("x1").Base(), build.GP64())}
	x2 := operand.Mem{Base: build.Load(build.Param("x2").Base(), build.GP64())}
	n := build.Load(build.Param("x1").Len(), build.GP64())

	// Accumulation registers.

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []reg.VecVirtual{build.YMM(), build.YMM()}

	build.VZEROALL()

	bytesPerRegister := 32 // size of one YMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 4 64-bit values, or 8 32-bit values

	unrolls := []int{
		16 - len(acc), // all 16 XMM registers, minus the ones used for accumulation
		8,
		4,
		1,
	}

	for unrollIndex, unroll := range unrolls {
		build.Label(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex]))

		blockItems := itemsPerRegister * unroll
		blockBytesSize := bytesPerValue * blockItems

		build.CMPQ(n, operand.U32(blockItems))
		if unrollIndex < len(unrolls)-1 {
			build.JL(operand.LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex+1])))
		} else {
			build.JL(operand.LabelRef("tail"))
		}

		x1Regs := make([]reg.VecVirtual, unroll)
		for i := range x1Regs {
			x1Regs[i] = build.YMM()
		}

		for i, x1Reg := range x1Regs {
			VMOVUP[bits](x1.Offset(bytesPerRegister*i), x1Reg)
		}

		for i, x1Reg := range x1Regs {
			VFMADD231P[bits](x2.Offset(bytesPerRegister*i), x1Reg, acc[i%len(acc)])
		}

		build.ADDQ(operand.U32(blockBytesSize), x1.Base)
		build.ADDQ(operand.U32(blockBytesSize), x2.Base)
		build.SUBQ(operand.U32(blockItems), n)

		build.JMP(operand.LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	build.Label("tail")

	tail := build.XMM()
	VXORP[bits](tail, tail, tail)

	build.Label("tailLoop")

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("reduce"))

	x1Reg := build.XMM()
	VMOVS[bits](x1, x1Reg)
	VFMADD231S[bits](x2, x1Reg, tail)

	build.ADDQ(operand.U32(bits/8), x1.Base)
	build.ADDQ(operand.U32(bits/8), x2.Base)
	build.DECQ(n)

	build.JMP(operand.LabelRef("tailLoop"))

	// ---

	build.Label("reduce")

	for i := 1; i < len(acc); i++ {
		VADDP[bits](acc[0], acc[i], acc[0])
	}

	result := acc[0].AsX()

	top := build.XMM()
	build.VEXTRACTF128(operand.U8(1), acc[0], top)
	VADDP[bits](result, top, result)
	VADDP[bits](result, tail, result)

	switch bits {
	case 32:
		build.VHADDPS(result, result, result)
		build.VHADDPS(result, result, result)
	case 64:
		build.VHADDPD(result, result, result)
	default:
		panic(fmt.Errorf("unexpected bits %d", bits))
	}

	build.Store(result, build.ReturnIndex(0))

	build.RET()
}

func buildSSE(bits int) {
	name := "DotProdSSE"
	signature := fmt.Sprintf("func(x1, x2 []float%d) float%d", bits, bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s returns the dot product between x1 and x2 (%d bits, SSE required).", name, bits))

	x1 := operand.Mem{Base: build.Load(build.Param("x1").Base(), build.GP64())}
	x2 := operand.Mem{Base: build.Load(build.Param("x2").Base(), build.GP64())}
	n := build.Load(build.Param("x1").Len(), build.GP64())

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []reg.VecVirtual{build.XMM(), build.XMM()}

	for _, reg := range acc {
		XORP[bits](reg, reg)
	}

	// x2 memory alignment

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("reduce"))

	x2StartByte := build.GP64()
	build.MOVQ(x2.Base, x2StartByte)
	build.ANDQ(operand.U32(15), x2StartByte)
	build.JZ(operand.LabelRef("unrolledLoops"))

	switch bits {
	case 32:
		shifts := x2StartByte
		// 4 - floor(x2StartByte % 16 / 4)
		build.XORQ(operand.U32(15), shifts)
		build.INCQ(shifts)
		build.SHRQ(operand.U8(2), shifts)

		build.Label("alignmentLoop")

		reg := build.XMM()
		MOVS[bits](x1, reg)
		MULS[bits](x2, reg)
		ADDS[bits](reg, acc[0])

		build.ADDQ(operand.U32(bits/8), x1.Base)
		build.ADDQ(operand.U32(bits/8), x2.Base)
		build.DECQ(n)
		build.JZ(operand.LabelRef("reduce"))

		build.DECQ(shifts)
		build.JNZ(operand.LabelRef("alignmentLoop"))
	case 64:
		reg := build.XMM()
		MOVS[bits](x1, reg)
		MULS[bits](x2, reg)
		ADDS[bits](reg, acc[0])

		build.ADDQ(operand.U32(bits/8), x1.Base)
		build.ADDQ(operand.U32(bits/8), x2.Base)
		build.DECQ(n)
	default:
		panic(fmt.Errorf("unexpected bits %d", bits))
	}

	build.Label("unrolledLoops")

	bytesPerRegister := 16 // size of one XMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 2 64-bit values, or 4 32-bit values

	unrolls := []int{
		16 - len(acc), // all 16 XMM registers, minus the ones used for accumulation
		8,
		4,
		1,
	}

	for unrollIndex, unroll := range unrolls {
		build.Label(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex]))

		blockItems := itemsPerRegister * unroll
		blockBytesSize := bytesPerValue * blockItems

		build.CMPQ(n, operand.U32(blockItems))
		if unrollIndex < len(unrolls)-1 {
			build.JL(operand.LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex+1])))
		} else {
			build.JL(operand.LabelRef("tailLoop"))
		}

		xRegs := make([]reg.VecVirtual, unroll)
		for i := range xRegs {
			xRegs[i] = build.XMM()
		}

		for i, xReg := range xRegs {
			MOVUP[bits](x1.Offset(bytesPerRegister*i), xReg)
		}

		for i, xReg := range xRegs {
			MULP[bits](x2.Offset(bytesPerRegister*i), xReg)
		}

		for i, xReg := range xRegs {
			ADDP[bits](xReg, acc[i%len(acc)])
		}

		build.ADDQ(operand.U32(blockBytesSize), x1.Base)
		build.ADDQ(operand.U32(blockBytesSize), x2.Base)
		build.SUBQ(operand.U32(blockItems), n)

		build.JMP(operand.LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	build.Label("tailLoop")

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("reduce"))

	xReg := build.XMM()
	MOVS[bits](x1, xReg)
	MULS[bits](x2, xReg)
	ADDS[bits](xReg, acc[0])

	build.ADDQ(operand.U32(bits/8), x1.Base)
	build.ADDQ(operand.U32(bits/8), x2.Base)
	build.DECQ(n)

	build.JMP(operand.LabelRef("tailLoop"))

	// ---

	build.Label("reduce")

	result := acc[0]
	for i := 1; i < len(acc); i++ {
		ADDP[bits](acc[i], result)
	}

	switch bits {
	case 32:
		HADDP[bits](result, result)
		HADDP[bits](result, result)
	case 64:
		HADDP[bits](result, result)
	default:
		panic(fmt.Errorf("unexpected bits %d", bits))
	}

	build.Store(result, build.ReturnIndex(0))

	build.RET()
}
