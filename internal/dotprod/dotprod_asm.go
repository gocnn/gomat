package main

//go:generate go run . -out ../../mat/dotprod_amd64.s -stubs ../../mat/dotprod_amd64.go -pkg mat

import (
	"fmt"

	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

func main() {
	ConstraintExpr("amd64,gc,!purego")

	buildAVX(32)
	buildAVX(64)

	buildSSE(32)
	buildSSE(64)

	Generate()
}

type bitsToFuncOps map[int]func(...Op)
type bitsToFunc2Ops map[int]func(Op, Op)

var (
	ADDP       = bitsToFunc2Ops{32: ADDPS, 64: ADDPD}
	ADDS       = bitsToFunc2Ops{32: ADDSS, 64: ADDSD}
	HADDP      = bitsToFunc2Ops{32: HADDPS, 64: HADDPD}
	MOVS       = bitsToFunc2Ops{32: MOVSS, 64: MOVSD}
	MOVUP      = bitsToFunc2Ops{32: MOVUPS, 64: MOVUPD}
	MULP       = bitsToFunc2Ops{32: MULPS, 64: MULPD}
	MULS       = bitsToFunc2Ops{32: MULSS, 64: MULSD}
	VADDP      = bitsToFuncOps{32: VADDPS, 64: VADDPD}
	VFMADD231P = bitsToFuncOps{32: VFMADD231PS, 64: VFMADD231PD}
	VFMADD231S = bitsToFuncOps{32: VFMADD231SS, 64: VFMADD231SD}
	VMOVS      = bitsToFuncOps{32: VMOVSS, 64: VMOVSD}
	VMOVUP     = bitsToFuncOps{32: VMOVUPS, 64: VMOVUPD}
	VXORP      = bitsToFuncOps{32: VXORPS, 64: VXORPD}
	XORP       = bitsToFunc2Ops{32: XORPS, 64: XORPD}
)

func buildAVX(bits int) {
	name := fmt.Sprintf("DotProdAVX%d", bits)
	signature := fmt.Sprintf("func(x1, x2 []float%d) float%d", bits, bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s returns the dot product between x1 and x2 (%d bits, AVX required).", name, bits))

	x1 := Mem{Base: Load(Param("x1").Base(), GP64())}
	x2 := Mem{Base: Load(Param("x2").Base(), GP64())}
	n := Load(Param("x1").Len(), GP64())

	// Accumulation registers.

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []VecVirtual{YMM(), YMM()}

	VZEROALL()

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
		Label(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex]))

		blockItems := itemsPerRegister * unroll
		blockBytesSize := bytesPerValue * blockItems

		CMPQ(n, U32(blockItems))
		if unrollIndex < len(unrolls)-1 {
			JL(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex+1])))
		} else {
			JL(LabelRef("tail"))
		}

		x1Regs := make([]VecVirtual, unroll)
		for i := range x1Regs {
			x1Regs[i] = YMM()
		}

		for i, x1Reg := range x1Regs {
			VMOVUP[bits](x1.Offset(bytesPerRegister*i), x1Reg)
		}

		for i, x1Reg := range x1Regs {
			VFMADD231P[bits](x2.Offset(bytesPerRegister*i), x1Reg, acc[i%len(acc)])
		}

		ADDQ(U32(blockBytesSize), x1.Base)
		ADDQ(U32(blockBytesSize), x2.Base)
		SUBQ(U32(blockItems), n)

		JMP(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	Label("tail")

	tail := XMM()
	VXORP[bits](tail, tail, tail)

	Label("tailLoop")

	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	x1Reg := XMM()
	VMOVS[bits](x1, x1Reg)
	VFMADD231S[bits](x2, x1Reg, tail)

	ADDQ(U32(bits/8), x1.Base)
	ADDQ(U32(bits/8), x2.Base)
	DECQ(n)

	JMP(LabelRef("tailLoop"))

	// ---

	Label("reduce")

	for i := 1; i < len(acc); i++ {
		VADDP[bits](acc[0], acc[i], acc[0])
	}

	result := acc[0].AsX()

	top := XMM()
	VEXTRACTF128(U8(1), acc[0], top)
	VADDP[bits](result, top, result)
	VADDP[bits](result, tail, result)

	switch bits {
	case 32:
		VHADDPS(result, result, result)
		VHADDPS(result, result, result)
	case 64:
		VHADDPD(result, result, result)
	default:
		panic(fmt.Errorf("unexpected bits %d", bits))
	}

	Store(result, ReturnIndex(0))

	RET()
}

func buildSSE(bits int) {
	name := fmt.Sprintf("DotProdSSE%d", bits)
	signature := fmt.Sprintf("func(x1, x2 []float%d) float%d", bits, bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s returns the dot product between x1 and x2 (%d bits, SSE required).", name, bits))

	x1 := Mem{Base: Load(Param("x1").Base(), GP64())}
	x2 := Mem{Base: Load(Param("x2").Base(), GP64())}
	n := Load(Param("x1").Len(), GP64())

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []VecVirtual{XMM(), XMM()}

	for _, reg := range acc {
		XORP[bits](reg, reg)
	}

	// x2 memory alignment

	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	x2StartByte := GP64()
	MOVQ(x2.Base, x2StartByte)
	ANDQ(U32(15), x2StartByte)
	JZ(LabelRef("unrolledLoops"))

	switch bits {
	case 32:
		shifts := x2StartByte
		// 4 - floor(x2StartByte % 16 / 4)
		XORQ(U32(15), shifts)
		INCQ(shifts)
		SHRQ(U8(2), shifts)

		Label("alignmentLoop")

		reg := XMM()
		MOVS[bits](x1, reg)
		MULS[bits](x2, reg)
		ADDS[bits](reg, acc[0])

		ADDQ(U32(bits/8), x1.Base)
		ADDQ(U32(bits/8), x2.Base)
		DECQ(n)
		JZ(LabelRef("reduce"))

		DECQ(shifts)
		JNZ(LabelRef("alignmentLoop"))
	case 64:
		reg := XMM()
		MOVS[bits](x1, reg)
		MULS[bits](x2, reg)
		ADDS[bits](reg, acc[0])

		ADDQ(U32(bits/8), x1.Base)
		ADDQ(U32(bits/8), x2.Base)
		DECQ(n)
	default:
		panic(fmt.Errorf("unexpected bits %d", bits))
	}

	Label("unrolledLoops")

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
		Label(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex]))

		blockItems := itemsPerRegister * unroll
		blockBytesSize := bytesPerValue * blockItems

		CMPQ(n, U32(blockItems))
		if unrollIndex < len(unrolls)-1 {
			JL(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex+1])))
		} else {
			JL(LabelRef("tailLoop"))
		}

		xRegs := make([]VecVirtual, unroll)
		for i := range xRegs {
			xRegs[i] = XMM()
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

		ADDQ(U32(blockBytesSize), x1.Base)
		ADDQ(U32(blockBytesSize), x2.Base)
		SUBQ(U32(blockItems), n)

		JMP(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	Label("tailLoop")

	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	xReg := XMM()
	MOVS[bits](x1, xReg)
	MULS[bits](x2, xReg)
	ADDS[bits](xReg, acc[0])

	ADDQ(U32(bits/8), x1.Base)
	ADDQ(U32(bits/8), x2.Base)
	DECQ(n)

	JMP(LabelRef("tailLoop"))

	// ---

	Label("reduce")

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

	Store(result, ReturnIndex(0))

	RET()
}
