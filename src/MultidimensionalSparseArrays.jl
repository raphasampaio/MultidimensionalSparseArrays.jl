module MultidimensionalSparseArrays

include("sparse_array.jl")

export SparseArray, nnz, sparsity, stored_indices, stored_values, stored_pairs,
       spzeros, spones, spfill, findnz, dropstored!, compress!, hasindex, to_dense

end
