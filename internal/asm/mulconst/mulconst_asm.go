package main

//go:generate go run . -bits 64 -out ../../../asm/asm64/mulconst_amd64.s -stubs ../../../asm/asm64/mulconst_amd64.go -pkg asm64
//go:generate go run . -bits 32 -out ../../../asm/asm32/mulconst_amd64.s -stubs ../../../asm/asm32/mulconst_amd64.go -pkg asm32

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

var (
	MOVS        = map[int]func(operand.Op, operand.Op){32: build.MOVSS, 64: build.MOVSD}
	MOVUP       = map[int]func(operand.Op, operand.Op){32: build.MOVUPS, 64: build.MOVUPD}
	MULP        = map[int]func(operand.Op, operand.Op){32: build.MULPS, 64: build.MULPD}
	MULS        = map[int]func(operand.Op, operand.Op){32: build.MULSS, 64: build.MULSD}
	SHUFP       = map[int]func(operand.Op, operand.Op, operand.Op){32: build.SHUFPS, 64: build.SHUFPD}
	VBROADCASTS = map[int]func(...operand.Op){32: build.VBROADCASTSS, 64: build.VBROADCASTSD}
	VMOVUP      = map[int]func(...operand.Op){32: build.VMOVUPS, 64: build.VMOVUPD}
	VMULP       = map[int]func(...operand.Op){32: build.VMULPS, 64: build.VMULPD}

	unrolls = []int{14, 8, 4, 1}
)

func buildAVX(bits int) {
	name := "MulConstAVX"
	signature := fmt.Sprintf("func(c float%d, x, y []float%d)", bits, bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s multiplies each element of x by a constant value c, storing the result in y (%d bits, AVX required).", name, bits))

	c := build.Load(build.Param("c"), build.XMM())
	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}
	n := build.Load(build.Param("x").Len(), build.GP64())

	cy := build.YMM()
	VBROADCASTS[bits](c, cy)

	bytesPerRegister := 32 // size of one YMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 4 64-bit values, or 8 32-bit values

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

		regs := make([]reg.VecVirtual, unroll)
		for i := range regs {
			regs[i] = build.YMM()
		}

		for i, r := range regs {
			VMULP[bits](x.Offset(bytesPerRegister*i), cy, r)
		}
		for i, r := range regs {
			VMOVUP[bits](r, y.Offset(bytesPerRegister*i))
		}

		build.ADDQ(operand.U32(blockBytesSize), x.Base)
		build.ADDQ(operand.U32(blockBytesSize), y.Base)
		build.SUBQ(operand.U32(blockItems), n)

		build.JMP(operand.LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	build.Label("tailLoop")

	r := build.XMM()

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("end"))

	MOVS[bits](x, r)
	MULS[bits](c, r)
	MOVS[bits](r, y)

	build.ADDQ(operand.U32(bits/8), x.Base)
	build.ADDQ(operand.U32(bits/8), y.Base)
	build.DECQ(n)

	build.JMP(operand.LabelRef("tailLoop"))

	build.Label("end")

	build.RET()
}

func buildSSE(bits int) {
	name := "MulConstSSE"
	signature := fmt.Sprintf("func(c float%d, x, y []float%d)", bits, bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s multiplies each element of x by a constant value c, storing the result in y (%d bits, SSE required).", name, bits))

	c := build.Load(build.Param("c"), build.XMM())
	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	y := operand.Mem{Base: build.Load(build.Param("y").Base(), build.GP64())}
	n := build.Load(build.Param("x").Len(), build.GP64())

	SHUFP[bits](operand.U8(0), c, c)

	bytesPerRegister := 16 // size of one XMM register
	bytesPerValue := bits / 8
	itemsPerRegister := 8 * bytesPerRegister / bits // 2 64-bit values, or 4 32-bit values

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

		regs := make([]reg.VecVirtual, unroll)
		for i := range regs {
			regs[i] = build.XMM()
		}

		for i, r := range regs {
			MOVUP[bits](x.Offset(bytesPerRegister*i), r)
		}
		for _, r := range regs {
			MULP[bits](c, r)
		}
		for i, r := range regs {
			MOVUP[bits](r, y.Offset(bytesPerRegister*i))
		}

		build.ADDQ(operand.U32(blockBytesSize), x.Base)
		build.ADDQ(operand.U32(blockBytesSize), y.Base)
		build.SUBQ(operand.U32(blockItems), n)

		build.JMP(operand.LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	build.Label("tailLoop")

	r := build.XMM()

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("end"))

	MOVS[bits](x, r)
	MULS[bits](c, r)
	MOVS[bits](r, y)

	build.ADDQ(operand.U32(bits/8), x.Base)
	build.ADDQ(operand.U32(bits/8), y.Base)
	build.DECQ(n)

	build.JMP(operand.LabelRef("tailLoop"))

	build.Label("end")
	build.RET()
}
