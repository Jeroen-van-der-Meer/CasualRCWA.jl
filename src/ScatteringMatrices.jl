abstract type AbstractScatteringMatrix <: AbstractMatrix{ComplexF64} end

"""
    struct ScatteringMatrix

Scattering matrix (S-matrix) describing the reflection and transmission of a
periodic layer wedged in between two homogeneous media.

# Properties

- `S11::Matrix{ComplexF64}`: Reflection in the top medium.
- `S12::Matrix{ComplexF64}`: Transmission from the top to the bottom medium.
- `S21::Matrix{ComplexF64}`: Transmission from the bottom to the top medium.
- `S22::Matrix{ComplexF64}`: Reflection in the bottom medium.
"""
struct ScatteringMatrix <: AbstractScatteringMatrix
    S11::Matrix{ComplexF64}
    S12::Matrix{ComplexF64}
    S21::Matrix{ComplexF64}
    S22::Matrix{ComplexF64}
    
    function ScatteringMatrix(
        S11::AbstractMatrix{<:Number},
        S12::AbstractMatrix{<:Number},
        S21::AbstractMatrix{<:Number},
        S22::AbstractMatrix{<:Number}
    )
        s11 = size(S11)
        s12 = size(S12)
        s21 = size(S21)
        s22 = size(S22)
        @assert s11[1] == s12[1]
        @assert s21[1] == s22[1]
        @assert s11[2] == s21[2]
        @assert s12[2] == s22[2]
        return new(S11, S12, S21, S22)
    end
end

Base.size(S::ScatteringMatrix) = (
    size(S.S11, 1) + size(S.S21, 1),
    size(S.S11, 2) + size(S.S12, 2)
)

function Base.getindex(S::ScatteringMatrix, i::Integer, j::Integer)
    s11 = size(S.S11)
    if i > s11[1]
        if j > s11[2]
            return S.S22[i - s11[1], j - s11[2]]
        else
            return S.S21[i - s11[1], j]
        end
    else
        if j > s11[2]
            return S.S12[i, j - s11[2]]
        else
            return S.S11[i, j]
        end
    end
end

"""
    struct SymmetricScatteringMatrix

Scattering matrix of an RCWA layer wedged between identical homogeneous media on
both sides. By symmetry, the scattering matrix may be expressed as a block
matrix S = [R T; T R], where R and T are reflection and transmission matrices,
respectively.

# Properties

- `S11::Matrix{ComplexF64}`: Reflection in either homogeneous medium.
- `S12::Matrix{ComplexF64}`: Transmission from either homogeneous
  medium into the periodic layer.
"""
struct SymmetricScatteringMatrix <: AbstractScatteringMatrix
    S11::Matrix{ComplexF64}
    S12::Matrix{ComplexF64}

    function SymmetricScatteringMatrix(
        S11::AbstractMatrix{<:Number},
        S12::AbstractMatrix{<:Number}
    )
        @assert size(S11) == size(S12)
        return new(S11, S12)
    end
end

function Base.getproperty(S::SymmetricScatteringMatrix, v::Symbol)
    if v === :S21
        return Base.getfield(S, :S12)
    elseif v === :S22
        return Base.getfield(S, :S11)
    else
        return Base.getfield(S, v)
    end
end

Base.size(S::SymmetricScatteringMatrix) = (
    size(S.S11, 1) + size(S.S12, 1),
    size(S.S11, 2) + size(S.S12, 2)
)

function Base.getindex(S::SymmetricScatteringMatrix, i::Integer, j::Integer)
    s11 = size(S.S11)
    if i > s11[1]
        if j > s11[2]
            return S.S11[i - s11[1], j - s11[2]]
        else
            return S.S12[i - s11[1], j]
        end
    else
        if j > s11[2]
            return S.S12[i, j - s11[2]]
        else
            return S.S11[i, j]
        end
    end
end

"""
    function compute_global_scattering_matrix(
        top_modes, layer_modes, bottom_modes, wave_vectors;
        thickness
    )

Compute the global scattering matrix associated to a stack of RCWA layers.

# Arguments

- `top_modes::LayerModes`: Modes associated to the top homogeneous medium.
- `layer_modes::AbstractVector{LayerModes}`: Modes associated to the RCWA
  layers.
- `bottom_modes::LayerModes`: Modes associated to the bottom homogeneous medium.
- `wavelength::Real`: Wavelength of the incoming light.
- `layer_thicknesses::AbstractVector{<:Real}`: Thicknesses of the RCWA layers.
"""
function compute_global_scattering_matrix(
    top_modes::LayerModes,
    layer_modes::AbstractVector{LayerModes},
    bottom_modes::LayerModes,
    free_space_modes::LayerModes,
    wavelength::Real,
    layer_thicknesses::AbstractVector{<:Real}
)
    nmodes = number_of_modes(top_modes) # = 2PQ
    @assert all(number_of_modes.(layer_modes) .== nmodes)
    @assert number_of_modes(bottom_modes) == nmodes
    @assert number_of_modes(free_space_modes) == nmodes
    @assert wavelength > 0
    nlayers = length(layer_modes)
    @assert length(layer_thicknesses) == nlayers

    Sg = compute_top_scattering_matrix(top_modes, free_space_modes)
    for (lt, lm) in zip(layer_thicknesses, layer_modes)
        S = compute_symmetric_scattering_matrix(lm, free_space_modes, wavelength, lt)
        Sg = star_product(Sg, S)
    end
    S = compute_bottom_scattering_matrix(free_space_modes, bottom_modes)
    Sg = star_product(Sg, S)
    return Sg
end

"""
    function compute_symmetric_scattering_matrix

Compute the scattering matrix associated to an RCWA layer wedged between two
empty media.

# Arguments

- `layer_modes::LayerModes`: Modes associated to the RCWA layer.
- `wavelength::Real`: Wavelength of the incoming light.
- `layer_thickness::Real`: Thickness of the RCWA layer.
"""
function compute_symmetric_scattering_matrix(
    layer_modes::LayerModes,
    free_space_modes::LayerModes,
    wavelength::Real,
    layer_thickness::Real
)
    @assert wavelength > 0
    @assert layer_thickness > 0
    E_modes = layer_modes.electricModes \ free_space_modes.electricModes
    M_modes = layer_modes.magneticModes \ free_space_modes.magneticModes
    A = E_modes + M_modes
    B = E_modes - M_modes
    X = Diagonal(exp.((-2pi * layer_thickness / wavelength) * layer_modes.eigenvalues))

    R, T = _compute_symmetric_scattering_matrix(A, B, X)
    return SymmetricScatteringMatrix(R, T)
end

function _compute_symmetric_scattering_matrix(
    A::Matrix{ComplexF64},
    B::Matrix{ComplexF64},
    X::Diagonal{ComplexF64}
)
    XA = X * A
    XB = X * B
    
    AiXB = A \ XB
    
    R1 = A - XB * AiXB
    R2 = XB * (A \ XA) - B
    R = R1 \ R2

    T1 = A - XB * AiXB
    T2 = XA - XB * (A \ B)
    T = T1 \ T2

    return R, T
end

"""
    function compute_top_scattering_matrix(top_modes)

Compute the scattering matrix between top (reflective) homogeneous medium and an
empty medium. Equivalently, it may be viewed as a general scattering matrix in
the special case where the thickness of the RCWA layer is zero.

# Arguments

- `top_modes::LayerModes`: Modes associated to the top homogeneous medium.
"""
function compute_top_scattering_matrix(
    top_modes::LayerModes,
    free_space_modes::LayerModes
)
    E_modes = free_space_modes.electricModes \ top_modes.electricModes
    M_modes = free_space_modes.magneticModes \ top_modes.magneticModes
    A = E_modes + M_modes
    B = E_modes - M_modes

    S11, S12, S21, S22 = _compute_flat_scattering_matrix(A, B)
    return ScatteringMatrix(S11, S12, S21, S22)
end

"""
    function compute_bottom_scattering_matrix(bottom_modes)

Compute the scattering matrix between bottom (transmissive) homogeneous medium
and an empty medium. Equivalently, it may be viewed as a general scattering
matrix in the special case where the thickness of the RCWA layer is zero.

# Arguments

- `bottom_modes::LayerModes`: Modes associated to the bottom homogeneous medium.
"""
function compute_bottom_scattering_matrix(
    free_space_modes::LayerModes,
    bottom_modes::LayerModes
)
    E_modes = free_space_modes.electricModes \ bottom_modes.electricModes
    M_modes = free_space_modes.magneticModes \ bottom_modes.magneticModes

    A = E_modes + M_modes
    B = E_modes - M_modes

    S22, S21, S12, S11 = _compute_flat_scattering_matrix(A, B)
    return ScatteringMatrix(S11, S12, S21, S22)
end

function _compute_flat_scattering_matrix(
    A::Matrix{ComplexF64},
    B::Matrix{ComplexF64}
)
    # FIXME: I think A and B have a very simple structure always. Can probably
    # make use of that.
    Ai = inv(A)
    mAiB = -Ai * B
    S11 = mAiB
    S12 = 2 * Ai
    S21 = (A + B * mAiB) / 2
    S22 = B * Ai
    return S11, S12, S21, S22
end

"""
    function star_product(A, B)

Redheffer star product, which may be regarded as the 'plumbing' of a coupled
pair of scattering matrices.

# Arguments

- `A::AbstractScatteringMatrix`
- `B::AbstractScatteringMatrix`
"""
function star_product(A::AbstractScatteringMatrix, B::AbstractScatteringMatrix)
    A11 = A.S11; A12 = A.S12; A21 = A.S21; A22 = A.S22;
    B11 = B.S11; B12 = B.S12; B21 = B.S21; B22 = B.S22;
    sA2 = size(A21, 1)
    sB1 = size(B11, 1)
    @assert sA2 == sB1

    # Temporary values
    T12 = A12 / (I(sA2) - B11 * A22)
    T21 = B21 / (I(sB1) - A22 * B11)

    P11 = A11 + T12 * B11 * A21
    P12 =       T12 * B12
    P21 =       T21 * A21
    P22 = B22 + T21 * A22 * B12

    return ScatteringMatrix(P11, P12, P21, P22)
end