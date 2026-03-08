abstract type AbstractLayer end

"""
    struct Layer

Layer described in terms of complex electric permittivity and complex magnetic
permeability.

Note that we use the negative sign convention where a wave propagating in the
+z-direction is written as exp(-ikz). As such, the imaginary part of the complex
permittivity is negative.

# Properties

- `eps::Matrix{ComplexF64}`: Complex relative electric permittivity per pixel.
- `mu::Matrix{ComplexF64}`: Complex relative magnetic permeability per pixel.
"""
struct Layer <: AbstractLayer
    eps::Matrix{ComplexF64} # Complex relative electric permittivity
    mu::Matrix{ComplexF64} # Complex relative magnetic permeability

    function Layer(
        eps::AbstractMatrix{<:Number},
        mu::AbstractMatrix{<:Number}
    )
        @assert all(imag(eps) .<= 0)
        return new(eps, mu)
    end
end

"""
    function Layer(nk)

Describe a layer in terms of its complex refractive index.

This description may only be used for non-magnetic materials.

Note that we use the negative sign convention where a wave propagating in the
+z-direction is written as exp(-ikz). As such, the imaginary part of the complex
refractive index is negative.

# Arguments

- `nk::AbstractMatrix{<:Number}`: Complex refractive index per pixel.
"""
function Layer(nk::AbstractMatrix{<:Number})
    @assert all(imag(nk) .<= 0)
    eps = nk.^2
    mu = ones(size(eps))
    return Layer(eps, mu)
end

HomogeneousLayer(eps::Number, mu::Number) = Layer([eps;;], [mu;;])
HomogeneousLayer(nk::Number) = Layer([nk;;])

is_homogeneous(layer::Layer) = (length(unique(layer.eps)) == 1) &&
    (length(unique(layer.mu)) == 1)

"""
    struct ConvolvedLayer

Layer in which the electric permittivity ε and magnetic permeability μ have been
prepared for use in RCWA computation. Specifically, ε and μ have been Fourier
transformed, and its Fourier coefficient have been rearranged in a convolution
matrix format which is convenient for efficiently performing RCWA.

# Arguments

- `conv_eps::BTTB`: Convolved electric permittivity.
- `conv_mu::BTTB`: Convolved magnetic permeability.
"""
struct ConvolvedLayer <: AbstractLayer
    conv_eps::BTTB
    conv_mu::BTTB

    function ConvolvedLayer(
        conv_eps::AbstractMatrix{<:Number},
        conv_mu::AbstractMatrix{<:Number}
    )
        @assert size(conv_eps, 1) == size(conv_eps, 2) ==
            size(conv_mu, 1) == size(conv_mu, 2)
        return new(conv_eps, conv_mu)
    end
end

convolve(layer::ConvolvedLayer, ::Tuple{<:Integer, <:Integer}) = layer

"""
    convolve(layer, number_of_harmonics)

Apply a convolution to a given layer. Concretely, this amounts to taking the 2D
Fourier transform, and subsequently reordering the elements in a Toeplitz block
matrix form.

# Arguments

- `layer::Layer`: The layer you want to convolve.
- `number_of_harmonics::Tuple{<:Integer, <:Integer}`: Number of harmonics in X
  and Y, respectively.
"""
function convolve(
    layer::Layer,
    number_of_harmonics::Tuple{<:Integer, <:Integer}
)
    conv_eps = convolve(layer.eps, number_of_harmonics)
    conv_mu = convolve(layer.mu, number_of_harmonics)
    return ConvolvedLayer(conv_eps, conv_mu)
end

function convolve(
    M::Matrix{ComplexF64},
    number_of_harmonics::Tuple{<:Integer, <:Integer}
)
    P, Q = number_of_harmonics
    @assert (P >= 1) && (Q >= 1)
    if length(unique(M)) == 1
        # For a homogeneous layer, the convolve operation is independent of the
        # size of M, so we set M to be the minimum size to make the computation
        # go through.
        M = first(M) * ones(P, Q)
    else
        # For a non-homogeneous layer, we could 'stretch' our matrix until the
        # computation becomes valid, however it is likely that the user is
        # incorrectly using too low a resolution.
        @assert (P <= size(M, 1)) && (Q <= size(M, 2))
    end
    FM = fft(M)
    PQ = P * Q
    blocks = [_get_block(FM, i, P, PQ) for i = -Q+2:Q]
    vc = blocks[Q:2Q-1]
    vr = blocks[Q:-1:1]
    return BTTB(vc, vr)
end

function _get_block(M::Matrix{ComplexF64}, i::Int64, P::Int64, PQ::Int64)
    if i <= 0
        i += size(M, 2)
    end
    vc = M[1:P, i] ./ length(M)
    vr = Vector{ComplexF64}(undef, P)
    vr[1] = vc[1]
    vr[2:end] .= M[end:-1:end-P+2, i] ./ length(M)
    return Toeplitz{ComplexF64}(vc, vr)
end

is_homogeneous(layer::ConvolvedLayer) = _is_scaled_identity(layer.conv_eps) &&
    _is_scaled_identity(layer.conv_mu)

"""
Check whether a matrix is a scalar multiple of the identity matrix.
"""
function _is_scaled_identity(M::AbstractMatrix{ComplexF64}; atol=1e-10)
    n = size(M, 1)
    s = first(M)
    for j in 1:n, i in 1:n
        expected = (i == j) ? s : zero(ComplexF64)
        if abs(M[i, j] - expected) > atol
            return false
        end
    end
    return true
end
