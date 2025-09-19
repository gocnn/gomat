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
	{`"math"`, `math "github.com/gocnn/gomat/internal/math32"`, false},

	// BLAS parameter types
	{"blas.DrotmParams", "blas.SrotmParams", false},

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

	// Files to generate
	files := []string{"level1.go", "level2.go", "level3.go"}

	for _, file := range files {
		srcPath := filepath.Join(srcDir, file)
		dstPath := filepath.Join(dstDir, file)

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
