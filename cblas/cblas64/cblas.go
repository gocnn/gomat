package cblas64

/*
#cgo CFLAGS: -g -O2
#cgo windows LDFLAGS: -LC:/PROGRA~2/Intel/oneAPI/mkl/latest/lib -lmkl_rt
#cgo windows LDFLAGS: -LC:/openblas/lib -lopenblas
#cgo linux LDFLAGS: -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_rt
#cgo linux LDFLAGS: -L/usr/lib/x86_64-linux-gnu/openblas-pthread -lopenblas
#cgo linux LDFLAGS: -L/usr/lib/x86_64-linux-gnu -lopenblas
#cgo darwin LDFLAGS: -framework Accelerate
#cgo darwin LDFLAGS: -L/opt/homebrew/opt/openblas -lopenblas
*/
import "C"
