//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Replacement rules for converting blas64 to blas32
var replacements = []struct {
	pattern string
	replace string
	isRegex bool
}{
	// Package name (must be first)
	{"package blas64", "package blas32", false},

	// Import replacements (before other replacements)
	{`"github.com/gocnn/gomat/internal/mat/f64"`, `"github.com/gocnn/gomat/internal/mat/f32"`, false},
	{`"github.com/gocnn/gomat/cblas/cblas64"`, `"github.com/gocnn/gomat/cblas/cblas32"`, false},
	{`math "github.com/gocnn/gomat/internal/math32"`, `math "github.com/gocnn/gomat/internal/math32"`, false},
	{`"math"`, `math "github.com/gocnn/gomat/internal/math32"`, false},

	// BLAS parameter types
	{"blas.DrotmParams", "blas.SrotmParams", false},

	// CBLAS function calls
	{"cblas64.Axpy", "cblas32.Axpy", false},
	{"cblas64.Scal", "cblas32.Scal", false},
	{"cblas64.Copy", "cblas32.Copy", false},
	{"cblas64.Swap", "cblas32.Swap", false},
	{"cblas64.Dot", "cblas32.Dot", false},
	{"cblas64.Nrm2", "cblas32.Nrm2", false},
	{"cblas64.Asum", "cblas32.Asum", false},
	{"cblas64.Iamax", "cblas32.Iamax", false},
	{"cblas64.Rotg", "cblas32.Rotg", false},
	{"cblas64.Rot", "cblas32.Rot", false},
	{"cblas64.Rotmg", "cblas32.Rotmg", false},
	{"cblas64.Rotm", "cblas32.Rotm", false},
	{"cblas64.Gemv", "cblas32.Gemv", false},
	{"cblas64.Symv", "cblas32.Symv", false},
	{"cblas64.Trmv", "cblas32.Trmv", false},
	{"cblas64.Trsv", "cblas32.Trsv", false},
	{"cblas64.Ger", "cblas32.Ger", false},
	{"cblas64.Syr", "cblas32.Syr", false},
	{"cblas64.Syr2", "cblas32.Syr2", false},
	{"cblas64.Gbmv", "cblas32.Gbmv", false},
	{"cblas64.Sbmv", "cblas32.Sbmv", false},
	{"cblas64.Tbmv", "cblas32.Tbmv", false},
	{"cblas64.Tbsv", "cblas32.Tbsv", false},
	{"cblas64.Spmv", "cblas32.Spmv", false},
	{"cblas64.Tpmv", "cblas32.Tpmv", false},
	{"cblas64.Tpsv", "cblas32.Tpsv", false},
	{"cblas64.Spr", "cblas32.Spr", false},
	{"cblas64.Spr2", "cblas32.Spr2", false},
	{"cblas64.Gemm", "cblas32.Gemm", false},
	{"cblas64.Symm", "cblas32.Symm", false},
	{"cblas64.Trmm", "cblas32.Trmm", false},
	{"cblas64.Trsm", "cblas32.Trsm", false},
	{"cblas64.Syrk", "cblas32.Syrk", false},
	{"cblas64.Syr2k", "cblas32.Syr2k", false},

	// All f64 to f32 internal function calls
	{"f64.AxpyInc", "f32.AxpyInc", false},
	{"f64.AxpyUnitary", "f32.AxpyUnitary", false},
	{"f64.DotUnitary", "f32.DotUnitary", false},
	{"f64.DotInc", "f32.DotInc", false},
	{"f64.L2NormInc", "f32.L2NormInc", false},
	{"f64.L2NormUnitary", "f32.L2NormUnitary", false},
	{"f64.ScalInc", "f32.ScalInc", false},
	{"f64.ScalUnitary", "f32.ScalUnitary", false},
	{"f64.Ger", "f32.Ger", false},
	{"f64.GemvN", "f32.GemvN", false},
	{"f64.GemvT", "f32.GemvT", false},

	// Constants
	{"safmin = 0x1p-1022", "safmin = 0x1p-126", false},

	// Array type declarations
	{`\[4\]float64`, `[4]float32`, true},

	// Type replacements (must be last to avoid conflicts)
	{"float64", "float32", false},
}

func main() {
	// Source and destination directories
	srcDir := "blas64"
	dstDir := "blas32"

	// Files to generate (both pure Go and CBLAS versions)
	files := []string{"level1.go", "level2.go", "level3.go", "level1_c.go", "level2_c.go", "level3_c.go"}

	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(dstDir, 0755); err != nil {
		fmt.Printf("Error creating directory %s: %v\n", dstDir, err)
		return
	}

	for _, file := range files {
		srcPath := filepath.Join(srcDir, file)
		dstPath := filepath.Join(dstDir, file)

		// Check if source file exists
		if _, err := os.Stat(srcPath); os.IsNotExist(err) {
			fmt.Printf("Source file %s does not exist, skipping\n", srcPath)
			continue
		}

		fmt.Printf("Generating %s from %s\n", dstPath, srcPath)

		// Read source file
		content, err := os.ReadFile(srcPath)
		if err != nil {
			fmt.Printf("Error reading %s: %v\n", srcPath, err)
			continue
		}

		// Apply replacements in order
		result := string(content)
		for _, repl := range replacements {
			if repl.isRegex {
				// Use regex for patterns with special regex syntax
				re := regexp.MustCompile(repl.pattern)
				result = re.ReplaceAllString(result, repl.replace)
			} else {
				// Simple string replacement
				result = strings.ReplaceAll(result, repl.pattern, repl.replace)
			}
		}

		// Write destination file
		err = os.WriteFile(dstPath, []byte(result), 0644)
		if err != nil {
			fmt.Printf("Error writing %s: %v\n", dstPath, err)
			continue
		}

		fmt.Printf("Successfully generated %s\n", dstPath)
	}
}
