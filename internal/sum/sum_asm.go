package main

//go:generate go run . -out ../../mat/sum_amd64.s -stubs ../../mat/sum_amd64.go -pkg mat

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
	ADDP  = map[int]func(Op, Op){32: ADDPS, 64: ADDPD}
	ADDS  = map[int]func(Op, Op){32: ADDSS, 64: ADDSD}
	HADDP = map[int]func(Op, Op){32: HADDPS, 64: HADDPD}
	MOVS  = map[int]func(Op, Op){32: MOVSS, 64: MOVSD}
	VADDP = map[int]func(...Op){32: VADDPS, 64: VADDPD}
	VADDS = map[int]func(...Op){32: VADDSS, 64: VADDSD}
	VXORP = map[int]func(...Op){32: VXORPS, 64: VXORPD}
	XORP  = map[int]func(Op, Op){32: XORPS, 64: XORPD}
)

func buildAVX(bits int) {
	name := fmt.Sprintf("SumAVX%d", bits)
	signature := fmt.Sprintf("func(x []float%d) float%d", bits, bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s returns the sum of all values of x (%d bits, AVX required).", name, bits))

	x := Mem{Base: Load(Param("x").Base(), GP64())}
	n := Load(Param("x").Len(), GP64())

	// Accumulation registers.

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []VecVirtual{YMM(), YMM()}

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
		Label(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex]))

		blockItems := itemsPerRegister * unroll
		blockBytesSize := bytesPerValue * blockItems

		CMPQ(n, U32(blockItems))
		if unrollIndex < len(unrolls)-1 {
			JL(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex+1])))
		} else {
			JL(LabelRef("tail"))
		}

		for i := 0; i < unroll; i++ {
			VADDP[bits](x.Offset(bytesPerRegister*i), acc[i%len(acc)], acc[i%len(acc)])
		}

		ADDQ(U32(blockBytesSize), x.Base)
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

	VADDS[bits](x, tail, tail)

	ADDQ(U32(bits/8), x.Base)
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
	name := fmt.Sprintf("SumSSE%d", bits)
	signature := fmt.Sprintf("func(x []float%d) float%d", bits, bits)
	TEXT(name, NOSPLIT, signature)
	Pragma("noescape")
	Doc(fmt.Sprintf("%s returns the sum of all values of x (%d bits, SSE required).", name, bits))

	x := Mem{Base: Load(Param("x").Base(), GP64())}
	n := Load(Param("x").Len(), GP64())

	// Accumulation registers. One could be sufficient,
	// but alternating between two should improve pipelining.
	acc := []VecVirtual{XMM(), XMM()}

	for _, reg := range acc {
		XORP[bits](reg, reg)
	}

	// x memory alignment

	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	x2StartByte := GP64()
	MOVQ(x.Base, x2StartByte)
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
		MOVS[bits](x, reg)
		ADDS[bits](reg, acc[0])

		ADDQ(U32(bits/8), x.Base)
		DECQ(n)
		JZ(LabelRef("reduce"))

		DECQ(shifts)
		JNZ(LabelRef("alignmentLoop"))
	case 64:
		reg := XMM()
		MOVS[bits](x, reg)
		ADDS[bits](reg, acc[0])

		ADDQ(U32(bits/8), x.Base)
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

		for i := 0; i < unroll; i++ {
			ADDP[bits](x.Offset(bytesPerRegister*i), acc[i%len(acc)])
		}

		ADDQ(U32(blockBytesSize), x.Base)
		SUBQ(U32(blockItems), n)

		JMP(LabelRef(fmt.Sprintf("unrolledLoop%d", unrolls[unrollIndex])))
	}

	// ---

	Label("tailLoop")

	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	ADDS[bits](x, acc[0])

	ADDQ(U32(bits/8), x.Base)
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
