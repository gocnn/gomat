//go:build ignore

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// Replacement rules for converting cblas64 to cblas32
var replacements = []struct {
	pattern string
	replace string
	isRegex bool
}{
	// Package name (must be first)
	{"package cblas64", "package cblas32", false},

	// BLAS parameter types
	{"blas.DrotmParams", "blas.SrotmParams", false},

	// Special case for idamax function (must come before general cblas_d* rule)
	{"cblas_idamax", "cblas_isamax", false},

	// C function calls - convert cblas_d* to cblas_s*
	{`cblas_d([a-z]+)`, `cblas_s$1`, true},

	// C type conversions
	{"C.double", "C.float", false},
	{"(*C.double)", "(*C.float)", false},

	// Type replacements (must be last to avoid conflicts)
	{"float64", "float32", false},
}

func main() {
	// Source and destination directories
	srcDir := "cblas64"
	dstDir := "cblas32"

	// Files to generate - now includes cblas.go
	files := []string{"cblas.go", "level1.go", "level2.go", "level3.go"}

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

	fmt.Println("\nCode generation completed successfully!")
	fmt.Printf("Generated %d files in %s directory\n", len(files), dstDir)
}
