# MultidimensionalSparseArrays.jl

[![CI](https://github.com/raphasampaio/MultidimensionalSparseArrays.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/raphasampaio/MultidimensionalSparseArrays.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/raphasampaio/multidimensionalsparsearrays.jl/graph/badge.svg?token=I2kXECoZxZ)](https://codecov.io/github/raphasampaio/multidimensionalsparsearrays.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

A Julia package for multidimensional sparse arrays.

## Overview

`MultidimensionalSparseArrays.jl` provides a `SparseArray` type that efficiently stores and manipulates multidimensional arrays with a high proportion of zero elements. Unlike dense arrays, `SparseArray` only stores non-zero values, significantly reducing memory consumption for sparse data.

This package is designed to be a flexible and intuitive tool for scientific computing, data analysis, and any domain where large, sparse multidimensional data structures are common.

## Comparison with `SparseArrays.jl`

`MultidimensionalSparseArrays.jl` is designed to provide a flexible and easy-to-use interface for N-dimensional sparse arrays. While Julia's standard library `SparseArrays.jl` is highly optimized for 1-D and 2-D sparse arrays (vectors and matrices), `MultidimensionalSparseArrays.jl` offers a more general-purpose solution for higher-dimensional sparse data.

Key differences include:

- **Dimensionality:** `SparseArrays.jl` focuses on `SparseVector` and `SparseMatrixCSC`. `MultidimensionalSparseArrays.jl` is built from the ground up for N-dimensional arrays.
- **Storage Format:** `MultidimensionalSparseArrays.jl` uses a dictionary-based storage (`Dict{CartesianIndex{N}, T}`), which is flexible for arbitrary dimensions. `SparseArrays.jl` uses the more rigid but highly efficient Compressed Sparse Column (CSC) format for matrices.
- **Use Case:** If you are working with 1-D or 2-D sparse arrays and require high-performance linear algebra operations, `SparseArrays.jl` is the ideal choice. If you need to work with sparse data in three or more dimensions, `MultidimensionalSparseArrays.jl` provides a more natural and convenient API.

## Features

- **Memory Efficiency:** Only non-zero elements are stored, making it ideal for high-dimensional sparse data.
- **Arbitrary Dimensionality:** Create sparse arrays of any number of dimensions.
- **Intuitive Indexing:** Use standard Julia indexing to access and modify elements.
- **Comprehensive API:** A rich set of functions for creating, manipulating, and analyzing sparse arrays.
- **Interoperability:** Easily convert between `SparseArray` and dense `Array` types.
- **Arithmetic Operations:** Perform element-wise arithmetic (+, -, *) on sparse arrays.

## Installation

The package can be installed with the Julia package manager. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```julia
pkg> add MultidimensionalSparseArrays
```

## Usage

### Creating a SparseArray

You can create a `SparseArray` in several ways:

```julia
using MultidimensionalSparseArrays

# Create an empty 3x4x2 sparse array of Float64
A = SparseArray{Float64}(3, 4, 2)

# Create from a dense array (only non-zero elements are stored)
dense_array = [1 0 3; 0 0 0; 2 0 0]
B = SparseArray(dense_array)

# Create a sparse array of zeros
C = spzeros(Int, 5, 5)

# Create a sparse array of ones
D = spones(2, 2)
```

### Accessing and Modifying Elements

Use standard array indexing to get and set elements. Accessing an unset element will throw a `BoundsError`.

```julia
A = SparseArray{Float64}(3, 3)

# Set values
A[1, 1] = 10.0
A[3, 2] = -5.0

# Get values
value = A[1, 1]  # returns 10.0

# Check if an index has a stored value
hasindex(A, 1, 1)  # true
hasindex(A, 1, 2)  # false

# To get a value with a default for unset indices
get(A, (1, 2), 0.0) # returns 0.0
```

### Handling Sparsity

Several functions are provided to work with the sparse nature of the array:

```julia
# Get the number of non-zero elements
nnz(B)

# Get the sparsity of the array (fraction of zero elements)
sparsity(B)

# Get the stored indices and values
indices = stored_indices(B)
values = stored_values(B)
pairs = stored_pairs(B)

# Find the indices and values of non-zero elements
(I, V) = findnz(B)

# Remove stored zeros to save memory
compress!(B)
```

### Arithmetic Operations

Element-wise arithmetic operations are supported:

```julia
A = SparseArray([1 0; 0 4])
B = SparseArray([0 2; 3 0])

# Addition
C = A + B  # [1 2; 3 4]

# Subtraction
D = A - B  # [1 -2; -3 4]

# Scalar multiplication
E = A * 2  # [2 0; 0 8]
```

### Utility Functions

- `to_dense(A)`: Convert a `SparseArray` to a dense `Array`.
- `dropstored!(A, val)`: Remove all stored entries equal to `val`.
- `spzeros`, `spones`, `spfill`: Create sparse arrays with specific values.

## API Reference

The following are the key exports of the package:

- `SparseArray`: The multidimensional sparse array type.
- `nnz`: Get the number of non-zero elements.
- `sparsity`: Get the fraction of zero elements.
- `stored_indices`, `stored_values`, `stored_pairs`: Iterators for stored elements.
- `spzeros`, `spones`, `spfill`: Constructors for sparse arrays.
- `findnz`: Find non-zero elements.
- `dropstored!`, `compress!`: Memory management functions.
- `hasindex`: Check for a stored value at an index.
- `to_dense`: Convert to a dense array.

For more details, please refer to the docstrings of the individual functions.

## Code Formatting

This project uses `JuliaFormatter.jl` to ensure a consistent code style. To format your code locally, run the following command from the root of the project:

```bash
julia --project=format format/format.jl
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
