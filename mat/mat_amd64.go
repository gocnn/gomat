//go:build amd64 && gc && !purego

package mat

import "golang.org/x/sys/cpu"

var (
	hasAVX  = cpu.X86.HasAVX
	hasAVX2 = cpu.X86.HasAVX2
	hasFMA  = cpu.X86.HasFMA
)
