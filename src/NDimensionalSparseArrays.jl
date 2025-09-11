module NDimensionalSparseArrays

include("ndsparsearray.jl")

export NDSparseArray, nnz, sparsity, stored_indices, stored_values, stored_pairs,
    spzeros, spones, spfill, findnz, dropstored!, compress!, hasindex, to_dense

end
