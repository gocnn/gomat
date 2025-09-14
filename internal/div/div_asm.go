package main

//go:generate go run . -out ../../mat/div_amd64.s -stubs ../../mat/div_amd64.go -pkg mat

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

const unroll = 16 // number of XMM or YMM registers

var (
	DIVP   = map[int]func(Op, Op){32: DIVPS, 64: DIVPD}
	DIVS   = map[int]func(Op, Op){32: DIVSS, 64: DIVSD}
	MOVS   = map[int]func(Op, Op){32: MOVSS, 64: MOVSD}
	MOVUP  = map[int]func(Op, Op){32: MOVUPS, 64: MOVUPD}
	VDIVP  = map[int]func(...Op){32: VDIVPS, 64: VDIVPD}
	VMOVUP = map[int]func(...Op){32: VMOVUPS, 64: VMOVUPD}
)

func buildAVX(bits int) {
	name := fmt.Sprintf("DivAVX%d", bits)
	signature := fmt.Sprintf("func(x1, x2, y []float%d)", bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s divides x1 by x2, element-wise, storing the result in y (%d bits, AVX required).", name, bits))

	x1 := Mem{Base: Load(Param("x1").Base(), GP64())}
	x2 := Mem{Base: Load(Param("x2").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}
	n := Load(Param("x1").Len(), GP64())

	regs := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		regs[i] = YMM()
	}

	bytesPerRegister := 32 // size of one YMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 4 64-bit values, or 8 32-bit values

	Label("unrolledLoop")

	blockItems := itemsPerRegister * unroll
	blockBytesSize := bytesPerValue * blockItems

	CMPQ(n, U32(blockItems))
	JL(LabelRef("singleRegisterLoop"))

	for i, reg := range regs {
		VMOVUP[bits](x1.Offset(bytesPerRegister*i), reg)
	}
	for i, reg := range regs {
		VDIVP[bits](x2.Offset(bytesPerRegister*i), reg, reg)
	}
	for i, reg := range regs {
		VMOVUP[bits](reg, y.Offset(bytesPerRegister*i))
	}

	ADDQ(U32(blockBytesSize), x1.Base)
	ADDQ(U32(blockBytesSize), x2.Base)
	ADDQ(U32(blockBytesSize), y.Base)
	SUBQ(U32(blockItems), n)

	JMP(LabelRef("unrolledLoop"))

	// ---

	Label("singleRegisterLoop")

	blockItems = itemsPerRegister
	blockBytesSize = (bits / 8) * blockItems

	reg := regs[0]

	CMPQ(n, U32(blockItems))
	JL(LabelRef("tailLoop"))

	VMOVUP[bits](x1, reg)
	VDIVP[bits](x2, reg, reg)
	VMOVUP[bits](reg, y)

	ADDQ(U32(blockBytesSize), x1.Base)
	ADDQ(U32(blockBytesSize), x2.Base)
	ADDQ(U32(blockBytesSize), y.Base)
	SUBQ(U32(blockItems), n)

	JMP(LabelRef("singleRegisterLoop"))

	// ---

	Label("tailLoop")

	reg = XMM()

	CMPQ(n, U32(0))
	JE(LabelRef("end"))

	MOVS[bits](x1, reg)
	DIVS[bits](x2, reg)
	MOVS[bits](reg, y)

	ADDQ(U32(bits/8), x1.Base)
	ADDQ(U32(bits/8), x2.Base)
	ADDQ(U32(bits/8), y.Base)
	DECQ(n)

	JMP(LabelRef("tailLoop"))

	Label("end")
	RET()
}

func buildSSE(bits int) {
	name := fmt.Sprintf("DivSSE%d", bits)
	signature := fmt.Sprintf("func(x1, x2, y []float%d)", bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s divides x1 by x2, element-wise, storing the result in y (%d bits, SSE required).", name, bits))

	x1 := Mem{Base: Load(Param("x1").Base(), GP64())}
	x2 := Mem{Base: Load(Param("x2").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}
	n := Load(Param("x1").Len(), GP64())

	// x2 memory alignment

	CMPQ(n, U32(0))
	JE(LabelRef("end"))

	x2StartByte := GP64()
	MOVQ(x2.Base, x2StartByte)
	ANDQ(U32(15), x2StartByte)
	JZ(LabelRef("unrolledLoop"))

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
		DIVS[bits](x2, reg)
		MOVS[bits](reg, y)

		ADDQ(U32(bits/8), x1.Base)
		ADDQ(U32(bits/8), x2.Base)
		ADDQ(U32(bits/8), y.Base)
		DECQ(n)
		JZ(LabelRef("end"))

		DECQ(shifts)
		JNZ(LabelRef("alignmentLoop"))
	case 64:
		reg := XMM()

		MOVS[bits](x1, reg)
		DIVS[bits](x2, reg)
		MOVS[bits](reg, y)

		ADDQ(U32(bits/8), x1.Base)
		ADDQ(U32(bits/8), x2.Base)
		ADDQ(U32(bits/8), y.Base)
		DECQ(n)
	default:
		panic(fmt.Errorf("unexpected bits %d", bits))
	}

	regs := make([]VecVirtual, unroll)
	for i := 0; i < unroll; i++ {
		regs[i] = XMM()
	}

	bytesPerRegister := 16 // size of one XMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 2 64-bit values, or 4 32-bit values

	Label("unrolledLoop")

	blockItems := itemsPerRegister * unroll
	blockBytesSize := bytesPerValue * blockItems

	CMPQ(n, U32(blockItems))
	JL(LabelRef("singleRegisterLoop"))

	for i, reg := range regs {
		MOVUP[bits](x1.Offset(bytesPerRegister*i), reg)
	}
	for i, reg := range regs {
		DIVP[bits](x2.Offset(bytesPerRegister*i), reg)
	}
	for i, reg := range regs {
		MOVUP[bits](reg, y.Offset(bytesPerRegister*i))
	}

	ADDQ(U32(blockBytesSize), x1.Base)
	ADDQ(U32(blockBytesSize), x2.Base)
	ADDQ(U32(blockBytesSize), y.Base)
	SUBQ(U32(blockItems), n)

	JMP(LabelRef("unrolledLoop"))

	// ---

	Label("singleRegisterLoop")

	blockItems = itemsPerRegister
	blockBytesSize = (bits / 8) * blockItems

	reg := regs[0]

	CMPQ(n, U32(blockItems))
	JL(LabelRef("tailLoop"))

	MOVUP[bits](x1, reg)
	DIVP[bits](x2, reg)
	MOVUP[bits](reg, y)

	ADDQ(U32(blockBytesSize), x1.Base)
	ADDQ(U32(blockBytesSize), x2.Base)
	ADDQ(U32(blockBytesSize), y.Base)
	SUBQ(U32(blockItems), n)

	JMP(LabelRef("singleRegisterLoop"))

	// ---

	Label("tailLoop")

	reg = XMM()

	CMPQ(n, U32(0))
	JE(LabelRef("end"))

	MOVS[bits](x1, reg)
	DIVS[bits](x2, reg)
	MOVS[bits](reg, y)

	ADDQ(U32(bits/8), x1.Base)
	ADDQ(U32(bits/8), x2.Base)
	ADDQ(U32(bits/8), y.Base)
	DECQ(n)

	JMP(LabelRef("tailLoop"))

	Label("end")
	RET()
}
