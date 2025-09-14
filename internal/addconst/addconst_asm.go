package main

//go:generate go run . -out ../../mat/addconst_amd64.s -stubs ../../mat/addconst_amd64.go -pkg mat

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

var (
	ADDP        = map[int]func(Op, Op){32: ADDPS, 64: ADDPD}
	ADDS        = map[int]func(Op, Op){32: ADDSS, 64: ADDSD}
	MOVS        = map[int]func(Op, Op){32: MOVSS, 64: MOVSD}
	MOVUP       = map[int]func(Op, Op){32: MOVUPS, 64: MOVUPD}
	SHUFP       = map[int]func(Op, Op, Op){32: SHUFPS, 64: SHUFPD}
	VADDP       = map[int]func(...Op){32: VADDPS, 64: VADDPD}
	VBROADCASTS = map[int]func(...Op){32: VBROADCASTSS, 64: VBROADCASTSD}
	VMOVUP      = map[int]func(...Op){32: VMOVUPS, 64: VMOVUPD}

	unrolls = []int{14, 8, 4, 1}
)

func buildAVX(bits int) {
	name := fmt.Sprintf("AddConstAVX%d", bits)
	signature := fmt.Sprintf("func(c float%d, x, y []float%d)", bits, bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s adds a constant value c to each element of x, storing the result in y (%d bits, AVX required).", name, bits))

	c := Load(Param("c"), XMM())
	x := Mem{Base: Load(Param("x").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}
	n := Load(Param("x").Len(), GP64())

	cy := YMM()
	VBROADCASTS[bits](c, cy)

	bytesPerRegister := 32 // size of one YMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 4 64-bit values, or 8 32-bit values

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

		regs := make([]VecVirtual, unroll)
		for i := range regs {
			regs[i] = YMM()
		}

		for i, r := range regs {
			VADDP[bits](x.Offset(bytesPerRegister*i), cy, r)
		}
		for i, r := range regs {
			VMOVUP[bits](r, y.Offset(bytesPerRegister*i))
		}

		ADDQ(U32(blockBytesSize), x.Base)
		ADDQ(U32(blockBytesSize), y.Base)
		SUBQ(U32(blockItems), n)

		JMP(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	Label("tailLoop")

	r := XMM()

	CMPQ(n, U32(0))
	JE(LabelRef("end"))

	MOVS[bits](x, r)
	ADDS[bits](c, r)
	MOVS[bits](r, y)

	ADDQ(U32(bits/8), x.Base)
	ADDQ(U32(bits/8), y.Base)
	DECQ(n)

	JMP(LabelRef("tailLoop"))

	Label("end")

	RET()
}

func buildSSE(bits int) {
	name := fmt.Sprintf("AddConstSSE%d", bits)
	signature := fmt.Sprintf("func(c float%d, x, y []float%d)", bits, bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s adds a constant value c to each element of x, storing the result in y (%d bits, SSE required).", name, bits))

	c := Load(Param("c"), XMM())
	x := Mem{Base: Load(Param("x").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}
	n := Load(Param("x").Len(), GP64())

	SHUFP[bits](U8(0), c, c)

	bytesPerRegister := 16 // size of one XMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 2 64-bit values, or 4 32-bit values

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

		regs := make([]VecVirtual, unroll)
		for i := range regs {
			regs[i] = XMM()
		}

		for i, r := range regs {
			MOVUP[bits](x.Offset(bytesPerRegister*i), r)
		}
		for _, r := range regs {
			ADDP[bits](c, r)
		}
		for i, r := range regs {
			MOVUP[bits](r, y.Offset(bytesPerRegister*i))
		}

		ADDQ(U32(blockBytesSize), x.Base)
		ADDQ(U32(blockBytesSize), y.Base)
		SUBQ(U32(blockItems), n)

		JMP(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	Label("tailLoop")

	r := XMM()

	CMPQ(n, U32(0))
	JE(LabelRef("end"))

	MOVS[bits](x, r)
	ADDS[bits](c, r)
	MOVS[bits](r, y)

	ADDQ(U32(bits/8), x.Base)
	ADDQ(U32(bits/8), y.Base)
	DECQ(n)

	JMP(LabelRef("tailLoop"))

	Label("end")
	RET()
}
