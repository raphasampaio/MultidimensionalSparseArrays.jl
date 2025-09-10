"""
    SparseArray{T, N} <: AbstractArray{T, N}

A multidimensional sparse array that stores only explicitly set elements.
Accessing unset indices throws a BoundsError.

# Fields

  - `data::Dict{CartesianIndex{N}, T}`: Dictionary mapping indices to stored values
  - `dims::NTuple{N, Int}`: Dimensions of the array

# Examples

```julia
# Create a 3x3 sparse matrix
A = SparseArray{Float64, 2}((3, 3))
A[1, 1] = 5.0
A[2, 3] = 3.0

# Create from existing data (only non-zero elements stored)
B = SparseArray([1 0 3; 0 0 0; 2 0 0])
# B[1, 2] would throw BoundsError since it's unset
```
"""
struct SparseArray{T, N} <: AbstractArray{T, N}
    data::Dict{CartesianIndex{N}, T}
    dims::NTuple{N, Int}

    function SparseArray{T, N}(dims::NTuple{N, Int}) where {T, N}
        return new{T, N}(Dict{CartesianIndex{N}, T}(), dims)
    end
end

# Convenience constructors
SparseArray{T}(dims::NTuple{N, Int}) where {T, N} = SparseArray{T, N}(dims)

SparseArray{T}(dims::Vararg{Int, N}) where {T, N} = SparseArray{T, N}(dims)

# Array-like constructor: SparseArray{T, N}(undef, dims...)
SparseArray{T, N}(::UndefInitializer, dims::Vararg{Int, N}) where {T, N} =
    SparseArray{T, N}(dims)

SparseArray{T, N}(::UndefInitializer, dims::NTuple{N, Int}) where {T, N} =
    SparseArray{T, N}(dims)

# Constructor from dense array with optional tolerance for floating point
function SparseArray(A::AbstractArray{T, N}; atol::Real = 0) where {T, N}
    sparse_array = SparseArray{T, N}(size(A))
    zero_val = zero(T)
    
    for I in CartesianIndices(A)
        val = A[I]
        # Only store non-zero values (with tolerance for floating point)
        if T <: AbstractFloat
            if abs(val - zero_val) > atol
                sparse_array.data[I] = val
            end
        else
            if val != zero_val
                sparse_array.data[I] = val
            end
        end
    end
    return sparse_array
end

# Required AbstractArray interface
Base.size(A::SparseArray) = A.dims
Base.IndexStyle(::Type{<:SparseArray}) = IndexCartesian()

# Linear indexing support
@inline function Base.getindex(A::SparseArray, i::Int)
    @boundscheck checkbounds(A, i)
    idx = CartesianIndices(A)[i]
    haskey(A.data, idx) || throw(BoundsError(A, i))
    return A.data[idx]
end

@inline function Base.setindex!(A::SparseArray, val, i::Int)
    @boundscheck checkbounds(A, i)
    idx = CartesianIndices(A)[i]
    A.data[idx] = val
    return val
end

# Indexing
@inline function Base.getindex(A::SparseArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    idx = CartesianIndex(I)
    haskey(A.data, idx) || throw(BoundsError(A, I))
    return A.data[idx]
end

@inline function Base.getindex(A::SparseArray, I::CartesianIndex)
    @boundscheck checkbounds(A, I)
    haskey(A.data, I) || throw(BoundsError(A, I))
    return A.data[I]
end

@inline function Base.setindex!(A::SparseArray{T, N}, val, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    idx = CartesianIndex(I)
    A.data[idx] = val
    return val
end

@inline function Base.setindex!(A::SparseArray, val, I::CartesianIndex)
    @boundscheck checkbounds(A, I)
    A.data[I] = val
    return val
end

# Delete methods
"""
    delete!(A::SparseArray, I...)

Remove the stored value at index I. After deletion, accessing that index will throw BoundsError.
"""
function Base.delete!(A::SparseArray{T, N}, I::Vararg{Int, N}) where {T, N}
    @boundscheck checkbounds(A, I...)
    idx = CartesianIndex(I)
    delete!(A.data, idx)
    return A
end

function Base.delete!(A::SparseArray, I::CartesianIndex)
    @boundscheck checkbounds(A, I)
    delete!(A.data, I)
    return A
end

# Iteration - only iterates over stored values
function Base.iterate(A::SparseArray)
    iter_state = iterate(A.data)
    isnothing(iter_state) && return nothing
    (idx, val), state = iter_state
    return (val, state)
end

function Base.iterate(A::SparseArray, state)
    iter_state = iterate(A.data, state)
    isnothing(iter_state) && return nothing
    (idx, val), new_state = iter_state
    return (val, new_state)
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


# Display (basic version - improved version defined later)

# Basic arithmetic operations
Base.:(==)(A::SparseArray, B::SparseArray) =
    size(A) == size(B) && A.data == B.data

# Copy (more efficient)
function Base.copy(A::SparseArray{T, N}) where {T, N}
    B = SparseArray{T, N}(A.dims)
    merge!(B.data, A.data)
    return B
end

Base.similar(A::SparseArray{T, N}) where {T, N} =
    SparseArray{T, N}(A.dims)

Base.similar(A::SparseArray{T, N}, ::Type{S}) where {T, S, N} =
    SparseArray{S, N}(A.dims)

Base.similar(A::SparseArray, ::Type{S}, dims::Dims) where {S} =
    SparseArray{S, length(dims)}(dims)

# Specialized constructors
"""
    spzeros(T, dims...)

Create a sparse array of zeros with element type `T` and given dimensions.
"""
spzeros(::Type{T}, dims::Vararg{Int, N}) where {T, N} = SparseArray{T, N}(dims)
spzeros(::Type{T}, dims::NTuple{N, Int}) where {T, N} = SparseArray{T, N}(dims)
spzeros(dims::Vararg{Int, N}) where {N} = spzeros(Float64, dims...)

"""
    spones(T, dims...)

Create a sparse array filled with ones of type `T` and given dimensions.
Note: This creates a dense-like structure, which may not be memory efficient for large arrays.
"""
function spones(::Type{T}, dims::Vararg{Int, N}) where {T, N}
    A = SparseArray{T, N}(dims)
    one_val = one(T)
    for I in CartesianIndices(A)
        A.data[I] = one_val
    end
    return A
end
spones(dims::Vararg{Int, N}) where {N} = spones(Float64, dims...)

"""
    spfill(val, dims...)

Create a sparse array filled with the given value.
"""
function spfill(val::T, dims::Vararg{Int, N}) where {T, N}
    A = SparseArray{T, N}(dims)
    if val != zero(T)
        for I in CartesianIndices(A)
            A.data[I] = val
        end
    end
    return A
end

# Fill methods
"""
    fill!(A::SparseArray, val)

Fill sparse array `A` with value `val` at all positions.
This stores the value at every index within the array bounds.
"""
function Base.fill!(A::SparseArray, val)
    for I in CartesianIndices(A)
        A.data[I] = val
    end
    return A
end

# Finding functions
"""
    findnz(A::SparseArray)

Return the indices and values of the stored (non-zero) elements in `A`.
Returns `(I, V)` where `I` is a vector of `CartesianIndex` and `V` is a vector of values.
"""
function findnz(A::SparseArray{T, N}) where {T, N}
    indices = collect(keys(A.data))
    values = collect(vals for vals in Base.values(A.data))
    return (indices, values)
end

"""
    findall(f, A::SparseArray)

Find all stored indices where function `f` returns true.
Only searches among explicitly stored values.
"""
function Base.findall(f::F, A::SparseArray) where {F<:Function}
    result = CartesianIndex{ndims(A)}[]
    
    # Check only stored values
    for (idx, val) in A.data
        if f(val)
            push!(result, idx)
        end
    end
    
    return result
end

# Resolve ambiguity with Base.findall(pred::Base.Fix2{typeof(in)}, x::AbstractArray)
function Base.findall(pred::Base.Fix2{typeof(in)}, A::SparseArray)
    # Use Base's implementation by converting to the function form
    return findall(x -> pred(x), A)
end

# Arithmetic operations
"""
    +(A::SparseArray, B::SparseArray)

Element-wise addition of two sparse arrays.
"""
function Base.:+(A::SparseArray{T, N}, B::SparseArray{S, N}) where {T, S, N}
    size(A) == size(B) || throw(DimensionMismatch("Array dimensions must match"))
    
    R = promote_type(T, S)
    result = SparseArray{R, N}(size(A))
    
    # Add elements from A
    for (idx, val_a) in A.data
        if haskey(B.data, idx)
            # Both arrays have this index
            result.data[idx] = val_a + B.data[idx]
        else
            # Only A has this index (B is effectively zero here)
            result.data[idx] = val_a
        end
    end
    
    # Add elements from B that aren't in A
    for (idx, val_b) in B.data
        if !haskey(A.data, idx)
            # Only B has this index (A is effectively zero here)
            result.data[idx] = val_b
        end
    end
    
    return result
end

"""
    -(A::SparseArray, B::SparseArray)

Element-wise subtraction of two sparse arrays.
"""
function Base.:-(A::SparseArray{T, N}, B::SparseArray{S, N}) where {T, S, N}
    size(A) == size(B) || throw(DimensionMismatch("Array dimensions must match"))
    
    R = promote_type(T, S)
    result = SparseArray{R, N}(size(A))
    
    # Subtract elements
    for (idx, val_a) in A.data
        if haskey(B.data, idx)
            # Both arrays have this index
            new_val = val_a - B.data[idx]
            if new_val != zero(R)
                result.data[idx] = new_val
            end
        else
            # Only A has this index (B is effectively zero here)
            result.data[idx] = val_a
        end
    end
    
    # Handle elements only in B
    for (idx, val_b) in B.data
        if !haskey(A.data, idx)
            # Only B has this index (A is effectively zero here)
            new_val = zero(R) - val_b  # 0 - val_b
            if new_val != zero(R)
                result.data[idx] = new_val
            end
        end
    end
    
    return result
end

"""
    *(A::SparseArray, scalar)

Scalar multiplication of sparse array.
"""
function Base.:*(A::SparseArray{T, N}, scalar::Number) where {T, N}
    S = promote_type(T, typeof(scalar))
    result = SparseArray{S, N}(size(A))
    
    if scalar != 0
        for (idx, val) in A.data
            result.data[idx] = val * scalar
        end
    end
    
    return result
end

Base.:*(scalar::Number, A::SparseArray) = A * scalar

# Base show method (without MIME) - delegates to text/plain
function Base.show(io::IO, A::SparseArray{T, N}) where {T, N}
    show(io, MIME"text/plain"(), A)
end

# Improved display with better formatting
function Base.show(io::IO, ::MIME"text/plain", A::SparseArray{T, N}) where {T, N}
    compact = get(io, :compact, false)
    
    if compact
        print(io, "$(size(A)) SparseArray{$T, $N}")
        return
    end
    
    stored_count = nnz(A)
    total_elements = length(A)
    sparsity_pct = round(sparsity(A) * 100, digits=2)
    
    println(io, "$(size(A)) SparseArray{$T, $N} with $stored_count stored entries:")
    println(io, "  Sparsity: $sparsity_pct% ($(total_elements - stored_count) zeros)")
    
    if stored_count > 0
        # Show up to 10 entries, sorted by index
        sorted_pairs = sort(collect(stored_pairs(A)), by = x -> x[1])
        display_count = min(10, length(sorted_pairs))
        
        for i in 1:display_count
            idx, val = sorted_pairs[i]
            println(io, "  $idx  =>  $val")
        end
        
        if stored_count > 10
            println(io, "  ⋮")
            println(io, "  ($(stored_count - 10) more entries)")
        end
    end
end

# Memory efficiency method
"""
    dropstored!(A::SparseArray, val)

Remove all stored entries that equal `val` from the sparse array.
This can help reduce memory usage when entries become equal to the default value.
"""
function dropstored!(A::SparseArray, val)
    to_delete = CartesianIndex{ndims(A)}[]
    for (idx, stored_val) in A.data
        if stored_val == val
            push!(to_delete, idx)
        end
    end
    
    for idx in to_delete
        delete!(A.data, idx)
    end
    
    return A
end

"""
    compress!(A::SparseArray)

Remove stored entries that equal zero to reduce memory usage.
"""
compress!(A::SparseArray{T}) where {T} = dropstored!(A, zero(T))

# Additional utility methods for the new semantics
"""
    hasindex(A::SparseArray, I...)

Check if index I has a stored value in the sparse array.
"""
hasindex(A::SparseArray{T, N}, I::Vararg{Int, N}) where {T, N} = haskey(A.data, CartesianIndex(I))
hasindex(A::SparseArray, I::CartesianIndex) = haskey(A.data, I)

"""
    stored_indices(A::SparseArray)

Return an iterator over the indices that have stored values.
"""
stored_indices(A::SparseArray) = keys(A.data)

"""
    stored_values(A::SparseArray)

Return an iterator over the stored values.
"""
stored_values(A::SparseArray) = values(A.data)

"""
    stored_pairs(A::SparseArray)

Return an iterator over (index, value) pairs for stored elements.
"""
stored_pairs(A::SparseArray) = pairs(A.data)

"""
    to_dense(A::SparseArray{T}) where T

Convert sparse array to dense array, filling unset indices with zero(T).
"""
function to_dense(A::SparseArray{T}) where T
    dense = fill(zero(T), size(A))
    for (idx, val) in A.data
        dense[idx] = val
    end
    return dense
end

# Override collect to return only stored values
"""
    collect(A::SparseArray)

Collect only the stored values in the sparse array.
To get a dense representation, use `to_dense(A)`.
"""
Base.collect(A::SparseArray) = collect(stored_values(A))
