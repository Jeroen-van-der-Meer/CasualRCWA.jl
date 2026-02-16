"""
    struct BTTB

Square block Toeplitz matrix of square Toeplitz blocks.
"""
struct BTTB <: AbstractMatrix{ComplexF64}
    vc::Vector{Toeplitz{ComplexF64}}
    vr::Vector{Toeplitz{ComplexF64}}
    P::Int64
    Q::Int64

    function BTTB(
        vc::AbstractVector{<:AbstractToeplitz},
        vr::AbstractVector{<:AbstractToeplitz}
    )
        @assert length(vc) == length(vr)
        @assert all(size.(vc, 1) .== size.(vc, 2) .== size.(vr, 1) .== size.(vr, 2))
        @assert first(vc) == first(vr)
        P = size(first(vc), 1)
        Q = length(vc)
        return new(vc, vr, P, Q)
    end
end

Base.size(A::BTTB) = (A.P * A.Q, A.P * A.Q)

function Base.getindex(A::BTTB, i::Integer, j::Integer)
    P = A.P
    i_P = div(i - 1, P) + 1 # Which block?
    i_Q = i - (i_P - 1) * P # Which index within said block?
    j_P = div(j - 1, P) + 1 # Likewise for 2nd coordinate
    j_Q = j - (j_P - 1) * P
    d = i_P - j_P
    if d >= 0
        return A.vc[d + 1][i_Q, j_Q]
    else
        return A.vr[1 - d][i_Q, j_Q]
    end
end

# For now, we do the operations by simply turning these matrices into ordinary
# ones.