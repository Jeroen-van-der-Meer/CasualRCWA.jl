"""
    struct DiagonalBlocks

Square 2 x 2 block matrix whose blocks are diagonal:

    [Diagonal(a)  Diagonal(b)
     Diagonal(c)  Diagonal(d)]

In a homogeneous layer the spatial harmonics decouple and only the two
polarizations of each harmonic mix. Hence its mode matrices, and the scattering
matrices built from homogeneous layers only, take this form. Products, sums and
inverses of such matrices are again of this form and cost O(n) instead of O(n³).
"""
struct DiagonalBlocks <: AbstractMatrix{ComplexF64}
    a::Vector{ComplexF64}
    b::Vector{ComplexF64}
    c::Vector{ComplexF64}
    d::Vector{ComplexF64}

    function DiagonalBlocks(a, b, c, d)
        @assert length(a) == length(b) == length(c) == length(d)
        return new(a, b, c, d)
    end
end

function DiagonalBlocks(J::UniformScaling, n::Integer)
    return DiagonalBlocks(fill(J.λ, n), zeros(n), zeros(n), fill(J.λ, n))
end

Base.size(D::DiagonalBlocks) = (2length(D.a), 2length(D.a))

function Base.getindex(D::DiagonalBlocks, i::Integer, j::Integer)
    n = length(D.a)
    ii, jj = mod1(i, n), mod1(j, n)
    ii == jj || return zero(ComplexF64)
    block = i <= n ? (j <= n ? D.a : D.b) : (j <= n ? D.c : D.d)
    return block[ii]
end

function Base.Matrix(D::DiagonalBlocks)
    n = length(D.a)
    M = zeros(ComplexF64, 2n, 2n)
    for i in 1:n
        M[i, i] = D.a[i]
        M[i, n + i] = D.b[i]
        M[n + i, i] = D.c[i]
        M[n + i, n + i] = D.d[i]
    end
    return M
end

# Structured arithmetic.

Base.:-(D::DiagonalBlocks) = DiagonalBlocks(-D.a, -D.b, -D.c, -D.d)
Base.:*(s::Number, D::DiagonalBlocks) = DiagonalBlocks(s * D.a, s * D.b, s * D.c, s * D.d)
Base.:/(D::DiagonalBlocks, s::Number) = DiagonalBlocks(D.a / s, D.b / s, D.c / s, D.d / s)

function Base.:+(D::DiagonalBlocks, E::DiagonalBlocks)
    return DiagonalBlocks(D.a + E.a, D.b + E.b, D.c + E.c, D.d + E.d)
end

function Base.:-(D::DiagonalBlocks, E::DiagonalBlocks)
    return DiagonalBlocks(D.a - E.a, D.b - E.b, D.c - E.c, D.d - E.d)
end

function Base.:*(D::DiagonalBlocks, E::DiagonalBlocks)
    return DiagonalBlocks(
        D.a .* E.a .+ D.b .* E.c, D.a .* E.b .+ D.b .* E.d,
        D.c .* E.a .+ D.d .* E.c, D.c .* E.b .+ D.d .* E.d
    )
end

function Base.inv(D::DiagonalBlocks)
    det = D.a .* D.d .- D.b .* D.c
    return DiagonalBlocks(D.d ./ det, -D.b ./ det, -D.c ./ det, D.a ./ det)
end

Base.:\(D::DiagonalBlocks, E::DiagonalBlocks) = inv(D) * E
Base.:/(D::DiagonalBlocks, E::DiagonalBlocks) = D * inv(E)

function Base.:*(X::Diagonal, D::DiagonalBlocks)
    n = length(D.a)
    x1, x2 = view(X.diag, 1:n), view(X.diag, (n + 1):2n)
    return DiagonalBlocks(x1 .* D.a, x1 .* D.b, x2 .* D.c, x2 .* D.d)
end

function Base.:-(X::Diagonal, D::DiagonalBlocks)
    n = length(D.a)
    x1, x2 = view(X.diag, 1:n), view(X.diag, (n + 1):2n)
    return DiagonalBlocks(x1 .- D.a, -D.b, -D.c, x2 .- D.d)
end

# Mixed arithmetic with dense matrices. Multiplication by a DiagonalBlocks
# matrix only scales and adds rows or columns, at a cost of O(n²).

function Base.:*(D::DiagonalBlocks, M::AbstractMatrix)
    n = length(D.a)
    M1, M2 = M[1:n, :], M[(n + 1):2n, :]
    return [D.a .* M1 .+ D.b .* M2; D.c .* M1 .+ D.d .* M2]
end

function Base.:*(D::DiagonalBlocks, v::AbstractVector)
    n = length(D.a)
    v1, v2 = v[1:n], v[(n + 1):2n]
    return [D.a .* v1 .+ D.b .* v2; D.c .* v1 .+ D.d .* v2]
end

function Base.:*(M::AbstractMatrix, D::DiagonalBlocks)
    n = length(D.a)
    M1, M2 = M[:, 1:n], M[:, (n + 1):2n]
    return [M1 .* transpose(D.a) .+ M2 .* transpose(D.c) M1 .* transpose(D.b) .+ M2 .* transpose(D.d)]
end

# Disambiguate against the method above.
Base.:*(D::DiagonalBlocks, E::Diagonal) = D * DiagonalBlocks(E)

function DiagonalBlocks(X::Diagonal)
    n = size(X, 1) ÷ 2
    return DiagonalBlocks(X.diag[1:n], zeros(n), zeros(n), X.diag[(n + 1):2n])
end

function _add_diagonal_blocks!(M::Matrix{ComplexF64}, D::DiagonalBlocks, s::Number)
    n = length(D.a)
    for i in 1:n
        M[i, i] += s * D.a[i]
        M[i, n + i] += s * D.b[i]
        M[n + i, i] += s * D.c[i]
        M[n + i, n + i] += s * D.d[i]
    end
    return M
end

Base.:+(M::Matrix{ComplexF64}, D::DiagonalBlocks) = _add_diagonal_blocks!(copy(M), D, 1)
Base.:+(D::DiagonalBlocks, M::Matrix{ComplexF64}) = M + D
Base.:-(M::Matrix{ComplexF64}, D::DiagonalBlocks) = _add_diagonal_blocks!(copy(M), D, -1)
Base.:-(D::DiagonalBlocks, M::Matrix{ComplexF64}) = _add_diagonal_blocks!(-M, D, 1)

# Solving against a dense matrix: inv(M) * D is cheaper than a linear solve with
# a dense right-hand side, since the multiplication by D costs only O(n²).
Base.:\(M::Matrix{ComplexF64}, D::DiagonalBlocks) = inv(M) * D
Base.:\(D::DiagonalBlocks, M::Matrix{ComplexF64}) = inv(D) * M
Base.:/(M::Matrix{ComplexF64}, D::DiagonalBlocks) = M * inv(D)
Base.:/(D::DiagonalBlocks, M::Matrix{ComplexF64}) = D * inv(M)
