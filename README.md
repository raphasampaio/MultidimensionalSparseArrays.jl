# NDimensionalSparseArrays.jl

[![CI](https://github.com/JuliaSparse/NDimensionalSparseArrays.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/JuliaSparse/NDimensionalSparseArrays.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/github/JuliaSparse/NDimensionalSparseArrays.jl/graph/badge.svg?token=I2kXECoZxZ)](https://codecov.io/github/JuliaSparse/NDimensionalSparseArrays.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

## Overview

`NDimensionalSparseArrays.jl` provides an efficient `NDSparseArray` type for working with sparse, n-dimensional arrays. It is designed to store and manipulate arrays with a high proportion of zero elements, reducing memory consumption significantly compared to dense arrays. The package is dependency-free, making it lightweight and easy to integrate into various projects.

## Features

- **Memory Efficiency**: Stores only non-zero elements, ideal for high-dimensional sparse data.
- **Arbitrary Dimensionality**: Handles sparse arrays in any number of dimensions.
- **No Dependencies**: The package is self-contained, with no external dependencies, ensuring a minimal footprint.
- **Intuitive Indexing**: Uses standard Julia array indexing for easy access and modification.
- **Flexible API**: A rich set of functions for creating, manipulating, and analyzing sparse arrays.
- **Interoperability**: Seamlessly convert between sparse and dense array types.

## Comparison with `SparseArrays.jl`

While `SparseArrays.jl` is highly optimized for 1D and 2D sparse arrays (vectors and matrices), `NDimensionalSparseArrays.jl` supports sparse data in **n**-dimensions, providing a more flexible and convenient solution for higher-dimensional use cases. Key differences include:

- **Dimensionality**: `SparseArrays.jl` is limited to 1D and 2D arrays, whereas `NDimensionalSparseArrays.jl` supports arbitrary-dimensional sparse arrays.
- **Storage Format**: Uses a dictionary-based storage format (`Dict{CartesianIndex{N}, T}`) suitable for any dimensionality, while `SparseArrays.jl` uses Compressed Sparse Column (CSC) format for matrices.
- **Use Case**: Best for working with higher-dimensional sparse arrays, while `SparseArrays.jl` excels in linear algebra operations for 1D and 2D sparse arrays.

## Installation

```julia
pkg> add NDimensionalSparseArrays
```

## Usage

### Creating a NDSparseArray

```julia
using NDimensionalSparseArrays

# Create an empty 3x4x2 sparse array of Float64
A = NDSparseArray{Float64}(3, 4, 2)

# Create from a dense array (only non-zero elements are stored)
dense_array = [1 0 3; 0 0 0; 2 0 0]
B = NDSparseArray(dense_array)

# Create a sparse array of zeros
C = spzeros(Int, 5, 5)

# Create a sparse array of ones
D = spones(2, 2)
```

### Accessing and Modifying Elements

```julia
A = NDSparseArray{Float64}(3, 3)

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

```julia
A = NDSparseArray([1 0; 0 4])
B = NDSparseArray([0 2; 3 0])

# Addition
C = A + B  # [1 2; 3 4]

# Subtraction
D = A - B  # [1 -2; -3 4]

# Scalar multiplication
E = A * 2  # [2 0; 0 8]
```

## API Reference

- `NDSparseArray`: The n-dimensional sparse array type.
- `nnz`: Get the number of non-zero elements.
- `sparsity`: Get the fraction of zero elements.
- `stored_indices`, `stored_values`, `stored_pairs`: Iterators for stored elements.
- `spzeros`, `spones`, `spfill`: Constructors for sparse arrays.
- `findnz`: Find non-zero elements.
- `dropstored!`, `compress!`: Memory management functions.
- `hasindex`: Check for a stored value at an index.
- `to_dense`: Convert to a dense array.

## Contributing

Contributions, bug reports, and feature requests are welcome! Feel free to open an issue or submit a pull request.
