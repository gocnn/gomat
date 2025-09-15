package main

//go:generate go run . -bits 64 -out ../../../asm/asm64/sum_amd64.s -stubs ../../../asm/asm64/sum_amd64.go -pkg asm64
//go:generate go run . -bits 32 -out ../../../asm/asm32/sum_amd64.s -stubs ../../../asm/asm32/sum_amd64.go -pkg asm32

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
	ADDP  = map[int]func(operand.Op, operand.Op){32: build.ADDPS, 64: build.ADDPD}
	ADDS  = map[int]func(operand.Op, operand.Op){32: build.ADDSS, 64: build.ADDSD}
	HADDP = map[int]func(operand.Op, operand.Op){32: build.HADDPS, 64: build.HADDPD}
	MOVS  = map[int]func(operand.Op, operand.Op){32: build.MOVSS, 64: build.MOVSD}
	VADDP = map[int]func(...operand.Op){32: build.VADDPS, 64: build.VADDPD}
	VADDS = map[int]func(...operand.Op){32: build.VADDSS, 64: build.VADDSD}
	VXORP = map[int]func(...operand.Op){32: build.VXORPS, 64: build.VXORPD}
	XORP  = map[int]func(operand.Op, operand.Op){32: build.XORPS, 64: build.XORPD}
)

func buildAVX(bits int) {
	name := "SumAVX"
	signature := fmt.Sprintf("func(x []float%d) float%d", bits, bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s returns the sum of all values of x (%d bits, AVX required).", name, bits))

	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	n := build.Load(build.Param("x").Len(), build.GP64())

	// Accumulation registers.

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []reg.VecVirtual{build.YMM(), build.YMM()}

	for _, r := range acc {
		VXORP[bits](r, r, r)
	}

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

		for i := 0; i < unroll; i++ {
			VADDP[bits](x.Offset(bytesPerRegister*i), acc[i%len(acc)], acc[i%len(acc)])
		}

		build.ADDQ(operand.U32(blockBytesSize), x.Base)
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

	VADDS[bits](x, tail, tail)

	build.ADDQ(operand.U32(bits/8), x.Base)
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
	name := "SumSSE"
	signature := fmt.Sprintf("func(x []float%d) float%d", bits, bits)
	build.TEXT(name, build.NOSPLIT, signature)
	build.Pragma("noescape")
	build.Doc(fmt.Sprintf("%s returns the sum of all values of x (%d bits, SSE required).", name, bits))

	x := operand.Mem{Base: build.Load(build.Param("x").Base(), build.GP64())}
	n := build.Load(build.Param("x").Len(), build.GP64())

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []reg.VecVirtual{build.XMM(), build.XMM()}

	for _, reg := range acc {
		XORP[bits](reg, reg)
	}

	// x memory alignment

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("reduce"))

	x2StartByte := build.GP64()
	build.MOVQ(x.Base, x2StartByte)
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
		MOVS[bits](x, reg)
		ADDS[bits](reg, acc[0])

		build.ADDQ(operand.U32(bits/8), x.Base)
		build.DECQ(n)
		build.JZ(operand.LabelRef("reduce"))

		build.DECQ(shifts)
		build.JNZ(operand.LabelRef("alignmentLoop"))
	case 64:
		reg := build.XMM()
		MOVS[bits](x, reg)
		ADDS[bits](reg, acc[0])

		build.ADDQ(operand.U32(bits/8), x.Base)
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

		for i := 0; i < unroll; i++ {
			ADDP[bits](x.Offset(bytesPerRegister*i), acc[i%len(acc)])
		}

		build.ADDQ(operand.U32(blockBytesSize), x.Base)
		build.SUBQ(operand.U32(blockItems), n)

		build.JMP(operand.LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	build.Label("tailLoop")

	build.CMPQ(n, operand.U32(0))
	build.JE(operand.LabelRef("reduce"))

	ADDS[bits](x, acc[0])

	build.ADDQ(operand.U32(bits/8), x.Base)
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
