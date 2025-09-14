//go:build amd64 && gc && !noasm && !gccgo

package mat

import "golang.org/x/sys/cpu"

var (
	hasFMA  = cpu.X86.HasFMA
	hasAVX  = cpu.X86.HasAVX
	hasAVX2 = cpu.X86.HasAVX2
)
