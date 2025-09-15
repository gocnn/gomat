package main

//go:generate go run . -bits 64 -out ../../../asm/asm64/add_amd64.s -stubs ../../../asm/asm64/add_amd64.go -pkg asm64
//go:generate go run . -bits 32 -out ../../../asm/asm32/add_amd64.s -stubs ../../../asm/asm32/add_amd64.go -pkg asm32

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

const unroll = 16 // number of XMM or YMM registers

type bitsToFuncOps map[int]func(...operand.Op)
type bitsToFunc2Ops map[int]func(operand.Op, operand.Op)

var (
	ADDP   = bitsToFunc2Ops{32: build.ADDPS, 64: build.ADDPD}
	ADDS   = bitsToFunc2Ops{32: build.ADDSS, 64: build.ADDSD}
	MOVS   = bitsToFunc2Ops{32: build.MOVSS, 64: build.MOVSD}
	MOVUP  = bitsToFunc2Ops{32: build.MOVUPS, 64: build.MOVUPD}
	VADDP  = bitsToFuncOps{32: build.VADDPS, 64: build.VADDPD}
	VADDS  = bitsToFuncOps{32: build.VADDSS, 64: build.VADDSD}
	VMOVS  = bitsToFuncOps{32: build.VMOVSS, 64: build.VMOVSD}
	VMOVUP = bitsToFuncOps{32: build.VMOVUPS, 64: build.VMOVUPD}
)

func buildAVX(bits int) {
	name := "AddAVX"
	signature := fmt.Sprintf("func(x1, x2, y []float%d)", bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s adds x1 and x2 element-wise, storing the result in y (%d bits, AVX required).", name, bits))

	x1 := operand.Mem{Base: build.Load(build.Param("x1").Base(), build.GP64())}
	x2 := operand.Mem{Base: build.Load(build.Param("x2").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}
	n := build.Load(build.Param("x1").Len(), build.GP64())

	regs := make([]reg.VecVirtual, unroll)
	for i := range unroll {
		regs[i] = build.YMM()
	}

	bytesPerRegister := 32 // size of one YMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 4 64-bit values, or 8 32-bit values

	build.Label("unrolledLoop")

	blockItems := itemsPerRegister * unroll
	blockBytesSize := bytesPerValue * blockItems

	build.CMPQ(n, operand.U32(blockItems))
	build.JL(operand.LabelRef("singleRegisterLoop"))

	for i, reg := range regs {
		VMOVUP[bits](x1.Offset(bytesPerRegister*i), reg)
	}

	for i, reg := range regs {
		VADDP[bits](x2.Offset(bytesPerRegister*i), reg, reg)
	}

	for i, reg := range regs {
		VMOVUP[bits](reg, y.Offset(bytesPerRegister*i))
	}

	build.ADDQ(operand.U32(blockBytesSize), x1.Base)
	build.ADDQ(operand.U32(blockBytesSize), x2.Base)
	build.ADDQ(operand.U32(blockBytesSize), y.Base)
	build.SUBQ(operand.U32(blockItems), n)

	build.JMP(operand.LabelRef("unrolledLoop"))

	// ---

	build.Label("singleRegisterLoop")

	blockItems = itemsPerRegister
	blockBytesSize = (bits / 8) * blockItems

	reg := regs[0]

	build.CMPQ(n, operand.U32(blockItems))
	build.JL(operand.LabelRef("tailLoop"))

	VMOVUP[bits](x1, reg)
	VADDP[bits](x2, reg, reg)
	VMOVUP[bits](reg, y)

	build.ADDQ(operand.U32(blockBytesSize), x1.Base)
	build.ADDQ(operand.U32(blockBytesSize), x2.Base)
	build.ADDQ(operand.U32(blockBytesSize), y.Base)
	build.SUBQ(operand.U32(blockItems), n)

	build.JMP(operand.LabelRef("singleRegisterLoop"))

	// ---

	build.Label("tailLoop")

	reg = build.XMM()

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("end"))

	VMOVS[bits](x1, reg)
	VADDS[bits](x2, reg, reg)
	VMOVS[bits](reg, y)

	build.ADDQ(operand.U32(bits/8), x1.Base)
	build.ADDQ(operand.U32(bits/8), x2.Base)
	build.ADDQ(operand.U32(bits/8), y.Base)
	build.DECQ(n)

	build.JMP(operand.LabelRef("tailLoop"))

	build.Label("end")
	build.RET()
}

func buildSSE(bits int) {
	name := "AddSSE"
	signature := fmt.Sprintf("func(x1, x2, y []float%d)", bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s adds x1 and x2 element-wise, storing the result in y (%d bits, SSE required).", name, bits))

	x1 := operand.Mem{Base: build.Load(build.Param("x1").Base(), build.GP64())}
	x2 := operand.Mem{Base: build.Load(build.Param("x2").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}
	n := build.Load(build.Param("x1").Len(), build.GP64())

	// x2 memory alignment

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("end"))

	x2StartByte := build.GP64()
	build.MOVQ(x2.Base, x2StartByte)
	build.ANDQ(operand.U32(15), x2StartByte)
	build.JZ(operand.LabelRef("unrolledLoop"))

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
		ADDS[bits](x2, reg)
		MOVS[bits](reg, y)

		build.ADDQ(operand.U32(bits/8), x1.Base)
		build.ADDQ(operand.U32(bits/8), x2.Base)
		build.ADDQ(operand.U32(bits/8), y.Base)
		build.DECQ(n)
		build.JZ(operand.LabelRef("end"))

		build.DECQ(shifts)
		build.JNZ(operand.LabelRef("alignmentLoop"))
	case 64:
		reg := build.XMM()

		MOVS[bits](x1, reg)
		ADDS[bits](x2, reg)
		MOVS[bits](reg, y)

		build.ADDQ(operand.U32(bits/8), x1.Base)
		build.ADDQ(operand.U32(bits/8), x2.Base)
		build.ADDQ(operand.U32(bits/8), y.Base)
		build.DECQ(n)
	default:
		panic(fmt.Errorf("unexpected bits %d", bits))
	}

	regs := make([]reg.VecVirtual, unroll)
	for i := range unroll {
		regs[i] = build.XMM()
	}

	bytesPerRegister := 16 // size of one XMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 2 64-bit values, or 4 32-bit values

	build.Label("unrolledLoop")

	blockItems := itemsPerRegister * unroll
	blockBytesSize := bytesPerValue * blockItems

	build.CMPQ(n, operand.U32(blockItems))
	build.JL(operand.LabelRef("singleRegisterLoop"))

	for i, reg := range regs {
		MOVUP[bits](x1.Offset(bytesPerRegister*i), reg)
	}

	for i, reg := range regs {
		ADDP[bits](x2.Offset(bytesPerRegister*i), reg)
	}

	for i, reg := range regs {
		MOVUP[bits](reg, y.Offset(bytesPerRegister*i))
	}

	build.ADDQ(operand.U32(blockBytesSize), x1.Base)
	build.ADDQ(operand.U32(blockBytesSize), x2.Base)
	build.ADDQ(operand.U32(blockBytesSize), y.Base)
	build.SUBQ(operand.U32(blockItems), n)

	build.JMP(operand.LabelRef("unrolledLoop"))

	// ---

	build.Label("singleRegisterLoop")

	blockItems = itemsPerRegister
	blockBytesSize = (bits / 8) * blockItems

	reg := regs[0]

	build.CMPQ(n, operand.U32(blockItems))
	build.JL(operand.LabelRef("tailLoop"))

	MOVUP[bits](x1, reg)
	ADDP[bits](x2, reg)
	MOVUP[bits](reg, y)

	build.ADDQ(operand.U32(blockBytesSize), x1.Base)
	build.ADDQ(operand.U32(blockBytesSize), x2.Base)
	build.ADDQ(operand.U32(blockBytesSize), y.Base)
	build.SUBQ(operand.U32(blockItems), n)

	build.JMP(operand.LabelRef("singleRegisterLoop"))

	// ---

	build.Label("tailLoop")

	reg = build.XMM()

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("end"))

	MOVS[bits](x1, reg)
	ADDS[bits](x2, reg)
	MOVS[bits](reg, y)

	build.ADDQ(operand.U32(bits/8), x1.Base)
	build.ADDQ(operand.U32(bits/8), x2.Base)
	build.ADDQ(operand.U32(bits/8), y.Base)
	build.DECQ(n)

	build.JMP(operand.LabelRef("tailLoop"))

	build.Label("end")
	build.RET()
}
