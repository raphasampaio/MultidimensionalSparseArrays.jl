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

# Array-like constructor: SparseArray{T, N}(undef, dims...)
SparseArray{T, N}(::UndefInitializer, dims::Vararg{Int, N}) where {T, N} =
    SparseArray{T, N}(dims, zero(T))

SparseArray{T, N}(::UndefInitializer, dims::NTuple{N, Int}) where {T, N} =
    SparseArray{T, N}(dims, zero(T))

# Constructor from dense array with optional tolerance for floating point
function SparseArray(A::AbstractArray{T, N}; atol::Real = 0) where {T, N}
    sparse_array = SparseArray{T, N}(size(A))
    default_val = sparse_array.default_value
    
    for I in CartesianIndices(A)
        val = A[I]
        # Use tolerance for floating point comparison
        if T <: AbstractFloat
            if abs(val - default_val) > atol
                sparse_array.data[I] = val
            end
        else
            if val != default_val
                sparse_array.data[I] = val
            end
        end
    end
    return sparse_array
end

# Required AbstractArray interface
Base.size(A::SparseArray) = A.dims
Base.IndexStyle(::Type{<:SparseArray}) = IndexLinear()

# Linear indexing support
@inline function Base.getindex(A::SparseArray, i::Int)
    @boundscheck checkbounds(A, i)
    idx = CartesianIndices(A)[i]
    return get(A.data, idx, A.default_value)
end

@inline function Base.setindex!(A::SparseArray, val, i::Int)
    @boundscheck checkbounds(A, i)
    idx = CartesianIndices(A)[i]
    if val == A.default_value
        delete!(A.data, idx)
    else
        A.data[idx] = val
    end
    return val
end

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

# Display (basic version - improved version defined later)

# Basic arithmetic operations
Base.:(==)(A::SparseArray, B::SparseArray) =
    size(A) == size(B) && A.default_value == B.default_value && A.data == B.data

# Copy (more efficient)
function Base.copy(A::SparseArray{T, N}) where {T, N}
    B = SparseArray{T, N}(A.dims, A.default_value)
    merge!(B.data, A.data)
    return B
end

Base.similar(A::SparseArray{T, N}) where {T, N} =
    SparseArray{T, N}(A.dims, A.default_value)

Base.similar(A::SparseArray{T, N}, ::Type{S}) where {T, S, N} =
    SparseArray{S, N}(A.dims, zero(S))

Base.similar(A::SparseArray, ::Type{S}, dims::Dims) where {S} =
    SparseArray{S, length(dims)}(dims, zero(S))

# Specialized constructors
"""
    spzeros(T, dims...)

Create a sparse array of zeros with element type `T` and given dimensions.
"""
spzeros(::Type{T}, dims::Vararg{Int, N}) where {T, N} = SparseArray{T, N}(dims, zero(T))
spzeros(::Type{T}, dims::NTuple{N, Int}) where {T, N} = SparseArray{T, N}(dims, zero(T))
spzeros(dims::Vararg{Int, N}) where {N} = spzeros(Float64, dims...)

"""
    spones(T, dims...)

Create a sparse array filled with ones of type `T` and given dimensions.
Note: This creates a dense-like structure, which may not be memory efficient for large arrays.
"""
function spones(::Type{T}, dims::Vararg{Int, N}) where {T, N}
    A = SparseArray{T, N}(dims, zero(T))
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
    A = SparseArray{T, N}(dims, zero(T))
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

Fill sparse array `A` with value `val`. If `val` is the default value, 
this efficiently clears all stored elements.
"""
function Base.fill!(A::SparseArray, val)
    if val == A.default_value
        empty!(A.data)
    else
        for I in CartesianIndices(A)
            A.data[I] = val
        end
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

Find all indices where function `f` returns true.
"""
function Base.findall(f::F, A::SparseArray) where {F<:Function}
    result = CartesianIndex{ndims(A)}[]
    
    # Check stored values
    for (idx, val) in A.data
        if f(val)
            push!(result, idx)
        end
    end
    
    # Check default values if predicate could match them
    if f(A.default_value)
        for I in CartesianIndices(A)
            if !haskey(A.data, I)
                push!(result, I)
            end
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
    result = SparseArray{R, N}(size(A), zero(R))
    
    # Add elements from A
    for (idx, val) in A.data
        result.data[idx] = val + B[idx]
    end
    
    # Add elements from B that aren't in A
    for (idx, val) in B.data
        if !haskey(A.data, idx)
            result.data[idx] = A[idx] + val
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
    result = SparseArray{R, N}(size(A), zero(R))
    
    # Subtract elements
    for (idx, val) in A.data
        new_val = val - B[idx]
        if new_val != zero(R)
            result.data[idx] = new_val
        end
    end
    
    # Handle elements only in B
    for (idx, val) in B.data
        if !haskey(A.data, idx)
            new_val = A[idx] - val
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
    result = SparseArray{S, N}(size(A), zero(S))
    
    if scalar != 0
        for (idx, val) in A.data
            result.data[idx] = val * scalar
        end
    end
    
    return result
end

Base.:*(scalar::Number, A::SparseArray) = A * scalar

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

Remove stored entries that equal the default value to reduce memory usage.
"""
compress!(A::SparseArray) = dropstored!(A, A.default_value)
