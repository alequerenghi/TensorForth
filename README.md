# TensorForth Interpreter

**TensorForth** is an interpreter for a high-performance, stack-based programming language designed to operate directly on 1D and 2D single-precision floating-point tensors (vectors and matrices). The execution architecture focuses on computational optimization and multi-core parallelization leveraging **OpenMP**, alongside efficient system-level operations like memory-mapped file I/O (`mmap`) and reference counting.

This project represents Part 1 (C Language Implementation) of the Advanced and Parallel Programming course.

---

## Key Features & Architecture

* 
**Postfix / Stack-Based Execution:** Programs are evaluated by reading tokens from left to right. Operands are pushed onto a global data stack, and postfix operators consume their respective parameters from the top of the stack.


* 
**OpenMP Parallelization:** Heavily demanding grid calculations—such as 2D convolutions and matrix/dot products—are parallelized across multiple CPU cores to maximize throughput.


* 
**Reference-Counted Memory Management:** Stack manipulation primitives like duplicate (`d`) and over (`o`) optimize performance by incrementing internal reference counters instead of triggering expensive deep memory copies. Buffers are safely deallocated only when their total reference count hits zero.


* 
**Zero-Copy Memory Mapping (`mmap`):** Native tensor storage reads utilize memory mapping (`mmap`) to project file streams directly into the process's virtual address space, eliminating standard kernel-to-user copy overhead and context switches.


* 
**Robust Exception Handling:** The interpreter ensures zero `Segmentation Fault` occurrences. Errors such as stack underflows, token type mismatches, and mathematical dimension incompatibilities are explicitly intercepted, logged, and trigger a graceful termination with a non-zero exit status.



---

## Project Structure & Compilation

The project uses an automated build configuration via a `Makefile` targeting modern Linux environments (such as Ubuntu).

### Build Steps

To compile the `tensorforth` binary with high-level compiler optimizations (`-O3`) and OpenMP enabled, invoke:

```bash
make

```

This command compiles the source and produces the final `tensorforth` executable.

### Command-Line Execution

To execute a program, pass the target source file path as an argument to the interpreter executable:

```bash
./tensorforth [path_to_source_file]

```

---

## Language Specifications

### 1. Basic Data Types

* 
**Tensors (1D & 2D):** Elements are strictly single-precision floating-point primitives (`float`). Inline tensor literals are represented inside square brackets separated exclusively by empty spaces (no commas).


* 
*Valid Syntax:* `[3.4 -8.53]` 


* 
*Invalid Syntax:* `[3.4, 8.5]` or `[3.4 -8.5  3]` 


* 
*Note:* Direct 2D matrix declaration is not natively supported by literal notation; matrices are initialized via a 1D vector and structured using the reshape (`r`) operator.




* 
**Strings:** Dedicated entirely to filesystem path mapping and declared using double quotes : `"input.pgm"`.



### 2. Stack Effect Notation

Operators are documented using standard Forth stack effect convention:

```text
( elements_before -- elements_after )

```

Items on the left are consumed from the stack prior to execution, and items on the right denote the resulting stack layout. The rightmost item represents the Top of Stack (TOS). For example, `( b a -- a+b )` indicates that `a` is on top of the stack, `b` is immediately beneath it, both are popped, and their element-wise sum is pushed.

---

## Comprehensive Operator Reference

### Stack Primitives (Reference-Counted)

| Operator | Stack Effect | Description |
| --- | --- | --- |
| `d` | `( a -- a a )` | <br>**Duplicate (dup):** Duplicates the top element by incrementing its internal reference counter without rewriting buffers.

 |
| `s` | `( b a -- a b )` | <br>**Swap:** Inverts the positions of the top two stack elements.

 |
| `o` | `( b a -- b a b )` | <br>**Over:** Clones the second element on the stack and pushes it to the top (updates reference counters).

 |
| `D` | `( a -- )` | **Drop:** Discards the top element. Reclaims memory allocation if reference counter equals 0.

 |

### Element-by-Element Arithmetic & Logic

All participating tensors must share identical dimensions.

| Operator | Stack Effect | Description |
| --- | --- | --- |
| `+` | `( b a -- a+b )` | Element-wise matrix/vector addition.

 |
| `-` | `( b a -- a-b )` | Element-wise matrix/vector subtraction.

 |
| `*` | `( b a -- a*b )` | Element-wise matrix/vector multiplication.

 |
| `<` | `( b a -- a<b )` | Element-wise less-than check. True evaluates to `1.0`, False to `0.0`.

 |
| `>` | `( b a -- a>b )` | Element-wise greater-than check. True evaluates to `1.0`, False to `0.0`.

 |
| `=` | `( b a -- a=b )` | Element-wise equality check. True evaluates to `1.0`, False to `0.0`.

 |
| `&` | `( b a -- a\&b )` | Element-wise logical AND. Operands must strictly contain only `0.0` and `1.0`.

 |
| `|` | `( b a -- a\/b )` | Element-wise logical OR. Operands must strictly contain only `0.0` and `1.0`.

 |
| `!` | `( a -- !a )` | Element-wise logical NOT. Operand values must strictly be `0.0` or `1.0`.

 |

### Advanced Tensor Operators

| Operator | Stack Effect | Description |
| --- | --- | --- |
| `$` | `( b a m -- m?a:b )` | **Selection Mask:** Evaluates condition mask `m` (composed of `0.0` and `1.0`). Pixels matching `1.0` are selected from `a`, otherwise from `b`.

 |
| `@` | `( b a -- a.b )` | <br>**Matrix / Dot Product:** Computes standard 2D matrix multiplication for matrices, or an inner dot product for 1D vectors. Parallelized with OpenMP.

 |
| `c` | `( a k -- conv(a,k) )` | <br>**2D Convolution:** Convolves input matrix `a` with a kernel matrix `k` using a stride of 1 and zero-padding. Output size matches `a`. Parallelized with OpenMP.

 |
| `r` | `( a s -- a' )` | <br>**Reshape:** Transforms tensor `a` to match shape parameters defined in vector `s`. Requires `prod(s) == len(a)` and uses the same memory space.

 |
| `_` | `( a -- a' )` | <br>**Ravel:** Flattens multi-dimensional matrix `a` into a continuous 1D vector without reallocation.

 |
| `#` | `( a -- #a )` | <br>**Shape:** Pushes a 1D vector containing the dimension lengths of tensor `a`.

 |
| `?` | `( s -- a )` | <br>**Random Generation:** Generates a tensor matching shape `s` initialized with random floating numbers in `[0, 1]`. Executed sequentially for `rand()` thread-safety.

 |
| `f` | `( s v -- a )` | <br>**Fill:** Creates a tensor matching shape `s` populated by cycling through elements in vector `v`. Useful for emulating element broadcasting.

 |
| `R` | `( a -- relu(a) )` | <br>**ReLU Activation:** Evaluates $max(0, x)$ element-by-element, clamping negative bounds to `0.0`.

 |
| `m` | `( b a -- min(a,b) )` | Returns the minimum element value comparing `a` and `b` element-by-element.

 |
| `M` | `( b a -- max(a,b) )` | Returns the maximum element value comparing `a` and `b` element-by-element.

 |
| `S` | `( a -- S(a) )` | <br>**Global Sum Reduction:** Accumulates all scalar floats inside `a` and returns a single-element 1D tensor.

 |

### Filesystem & Disk Image I/O

| Operator | Stack Effect | Description |
| --- | --- | --- |
| `P` | `( a -- )` | <br>**Print/Debug:** Serializes a tensor to stdout inside the mandatory tracking structure: `Tensor (shape=[shape], data=[data])` and pops it.

 |
| <br>`(` | `( filename -- tensor )` | **Read PGM Image:** Opens a binary grayscale PGM format image file, scaling unsigned char pixels `[0, 255]` into normalized floats `[0.0, 1.0]`.

 |
| `)` | `( a filename -- )` | <br>**Write PGM Image:** Saves 2D matrix `a` as a binary PGM image file. Values are clamped to `[0.0, 1.0]` and scaled back to `[0, 255]`.

 |
| `{` | `( filename -- tensor )` | <br>**Mmap Tensor:** Reads a native binary tensor file directly via `mmap` zero-copy system architectures.

 |
| `}` | `( a filename -- )` | <br>**Write Tensor:** Serializes tensor metadata and floating-point elements to disk.

 |

---

## On-Disk Tensor Format Specification

Native files saved or read via `{` and `}` are prefixed with a strict 64-byte structured metadata block followed immediately by the floating-point sequence:

```c
#define MAX_DIM 2

struct on_disk_tensor {
    int32_t shape[MAX_DIM];   // Dimensions sizes (fixed size of 2)
    int32_t ndim;             // Total active dimensions (1 or 2)
    off_t data_offset;        // Minimum structural byte offset for raw floats
};

```

To satisfy modern hardware access optimizations and ensure proper data alignment, `data_offset` is strictly fixed to **64 bytes**. The trailing unassigned section of the metadata block between parameters and the 64th byte acts as zero-filled structural padding.

---

## Functional Source Examples

### Example 1: Basic Vector Duplicate & Element Addition

```text
[5 5] d + P

```

1. 
`[5 5]` instantiates a 1D floating-point vector.


2. 
`d` duplicates the reference on the stack (reference counter increases; no allocations).


3. 
`+` pops both vectors, sums them element-wise, and pushes the result vector `[10 10]`.


4. 
`P` prints the calculated tensor output to stdout and drops it.



### Example 2: Matrix Multiplication

```text
[100 5] ? [300] ? [3 100] r @ P

```

1. Generates a random matrix of dimensions $100 \times 5$.


2. Allocates a random 1D vector containing 300 values.


3. Reshapes the 300-element vector into a structured $3 \times 100$ matrix.


4. Evaluates `@` to compute the matrix product of $(3 \times 100) \times (100 \times 5)$, pushing a $3 \times 5$ matrix onto the stack, which is then printed.
