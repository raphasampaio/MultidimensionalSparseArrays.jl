"""
    SparseArray{T, N} <: AbstractArray{T, N}

A multidimensional sparse array that stores only non-zero elements efficiently.

# Fields

  - `data::Dict{CartesianIndex{N}, T}`: Dictionary mapping indices to non-zero values
  - `dims::NTuple{N, Int}`: Dimensions of the array
  - `default_value::T`: Default value for unset indices (typically zero)

# Examples

```julia
# Create a 3x3 sparse matrix
A = SparseArray{Float64, 2}((3, 3))
A[1, 1] = 5.0
A[2, 3] = 3.0

# Create from existing data
B = SparseArray([1 0 3; 0 0 0; 2 0 0])
```
"""
struct SparseArray{T, N} <: AbstractArray{T, N}
    data::Dict{CartesianIndex{N}, T}
    dims::NTuple{N, Int}
    default_value::T

    function SparseArray{T, N}(dims::NTuple{N, Int}, default_value::T = zero(T)) where {T, N}
        return new{T, N}(Dict{CartesianIndex{N}, T}(), dims, default_value)
    end
end

# Convenience constructors
SparseArray{T}(dims::NTuple{N, Int}, default_value::T = zero(T)) where {T, N} =
    SparseArray{T, N}(dims, default_value)

SparseArray{T}(dims::Vararg{Int, N}) where {T, N} =
    SparseArray{T, N}(dims, zero(T))

# Constructor from dense array
function SparseArray(A::AbstractArray{T, N}) where {T, N}
    sparse_array = SparseArray{T, N}(size(A))
    for I in CartesianIndices(A)
        val = A[I]
        if val != sparse_array.default_value
            sparse_array.data[I] = val
        end
    end
    return sparse_array
end

# Required AbstractArray interface
Base.size(A::SparseArray) = A.dims
Base.IndexStyle(::Type{<:SparseArray}) = IndexCartesian()

# Indexing
@inline function Base.getindex(A::SparseArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    idx = CartesianIndex(I)
    return get(A.data, idx, A.default_value)
end

@inline function Base.getindex(A::SparseArray, I::CartesianIndex)
    @boundscheck checkbounds(A, I)
    return get(A.data, I, A.default_value)
end

@inline function Base.setindex!(A::SparseArray{T, N}, val, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    idx = CartesianIndex(I)
    if val == A.default_value
        delete!(A.data, idx)
    else
        A.data[idx] = val
    end
    return val
end

@inline function Base.setindex!(A::SparseArray, val, I::CartesianIndex)
    @boundscheck checkbounds(A, I)
    if val == A.default_value
        delete!(A.data, I)
    else
        A.data[I] = val
    end
    return val
end

# Iteration
function Base.iterate(A::SparseArray)
    iter_state = iterate(CartesianIndices(A))
    isnothing(iter_state) && return nothing
    idx, state = iter_state
    return (A[idx], state)
end

function Base.iterate(A::SparseArray, state)
    iter_state = iterate(CartesianIndices(A), state)
    isnothing(iter_state) && return nothing
    idx, new_state = iter_state
    return (A[idx], new_state)
end

# Additional useful methods
"""
    nnz(A::SparseArray)

Return the number of stored (non-zero) elements in the sparse array.
"""
nnz(A::SparseArray) = length(A.data)

"""
    sparsity(A::SparseArray)

Return the sparsity ratio (fraction of zero elements) of the array.
"""
sparsity(A::SparseArray) = 1.0 - nnz(A) / length(A)

"""
    stored_indices(A::SparseArray)

Return an iterator over the indices that have stored values.
"""
stored_indices(A::SparseArray) = keys(A.data)

"""
    stored_values(A::SparseArray)

Return an iterator over the stored non-zero values.
"""
stored_values(A::SparseArray) = values(A.data)

"""
    stored_pairs(A::SparseArray)

Return an iterator over (index, value) pairs for stored elements.
"""
stored_pairs(A::SparseArray) = pairs(A.data)

# Display
function Base.show(io::IO, ::MIME"text/plain", A::SparseArray{T, N}) where {T, N}
    println(io, "$(size(A)) SparseArray{$T, $N} with $(nnz(A)) stored entries:")
    if nnz(A) > 0
        for (idx, val) in stored_pairs(A)
            println(io, "  $idx  =>  $val")
        end
    end
end

# Basic arithmetic operations
Base.:(==)(A::SparseArray, B::SparseArray) =
    size(A) == size(B) && A.default_value == B.default_value && A.data == B.data

# Copy
function Base.copy(A::SparseArray{T, N}) where {T, N}
    B = SparseArray{T, N}(A.dims, A.default_value)
    for (k, v) in A.data
        B.data[k] = v
    end
    return B
end

Base.similar(A::SparseArray{T, N}) where {T, N} =
    SparseArray{T, N}(A.dims, A.default_value)

Base.similar(A::SparseArray{T, N}, ::Type{S}) where {T, S, N} =
    SparseArray{S, N}(A.dims, zero(S))
