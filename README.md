Overview
========
The repository would host questions and answers to:

1. TensorFlow and its backend to MLIR.
2. MLIR proper and how it handles StructuredOps.
3. MIOpen and how to get it integrated with MLIR proper.


TensorFlow and MLIR : infrastructure
====================================
- TF + MLIR build: how to enable it?
- How does MLIR get pulled in TF build? 
- MLIR-related tools for TF dialect?
- How to run TF dialect-related tests?
- We focused on `compiler/jit` before, what's the importance of `compiler/aot` with MLIR?
- How to enable `compiler/xla/service/mlir_gpu` backend in XLA?


TensorFlow and MLIR: operators / transoformations
=================================================
- How does HLO `Dot` gets lowered to LHLO `Dot` and further down to MLIR dialects such as Linalg or Affine?
- How does HLO `Conv` gets lowered to LHLO `Conv` and further down to MLIR dialects such as Linalg or Affine?
- How does decalarative optimization patterns of MLIR work in TensorFlow level?


MLIR
====
- How does StructuredOps work?
- Specifically, how to build an end-to-end example to depict some of the operations?
  - Fill
  - Copy
  - Matmul
  - Conv
- How could these ops be hooked with HIP/MIOpen?
- How does declarative patterns work?
  - Tiling
  - Fusion + tiling


MIOpen
======
- For implicit GEMM kernels, what parameters are tunable, and what parameters are to be supplied by clients?
- How to make implicit GEMM convolution kernels do regular GEMM operations?
- For tunable parameters, how/where to model tuning process?



