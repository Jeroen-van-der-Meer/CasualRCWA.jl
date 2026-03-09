"""
    struct Stack

A stack of layers sandwiched between two homogeneous half-spaces.

# Properties

- `topMedium::AbstractLayer`: Homogeneous top half-space (reflection region).
- `layers::Vector{<:AbstractLayer}`: Interior layers (may be patterned).
- `bottomMedium::AbstractLayer`: Homogeneous bottom half-space (transmission region).
- `thicknesses::Vector{Float64}`: Thickness of each interior layer.
- `period::Tuple{Float64, Float64}`: Unit cell size in (X, Y).
"""
struct Stack
    topMedium::AbstractLayer
    layers::Vector{<:AbstractLayer}
    bottomMedium::AbstractLayer
    thicknesses::Vector{Float64}
    period::Tuple{Float64, Float64}

    function Stack(
        top_medium::AbstractLayer,
        layers::AbstractVector{<:AbstractLayer},
        bottom_medium::AbstractLayer,
        thicknesses::AbstractVector{<:Real},
        period::Tuple{<:Real, <:Real}
    )
        @assert is_homogeneous(top_medium)
        @assert length(layers) == length(thicknesses)
        @assert is_homogeneous(bottom_medium)
        @assert (period[1] > 0) && (period[2] > 0)
        return new(top_medium, layers, bottom_medium, thicknesses, period)
    end
end

"""
    struct RCWASettings

All parameters needed to run an RCWA simulation.

# Properties

- `incomingWave::IncomingWave`: Incoming wave specification (angles, wavelength).
- `stack::Stack`: Layer stack including top and bottom half-spaces.
- `numberOfHarmonics::Tuple{Int64, Int64}`: Number of Fourier harmonics (P, Q).
"""
struct RCWASettings
    incomingWave::IncomingWave
    stack::Stack
    numberOfHarmonics::Tuple{Int64, Int64}
end

"""
    struct RCWAResult

Result of an RCWA simulation. Holds the global scattering matrix together with
all metadata needed to extract reflection/transmission coefficients and
diffraction efficiencies.

Use `reflection_coefficients`, `transmission_coefficients`, and
`diffraction_efficiencies` to query the result.
"""
struct RCWAResult
    scatteringMatrix::ScatteringMatrix
    waveVectors::PreparedWaveVectors
    input::RCWASettings
end

"""
    RCWA(settings::RCWASettings) -> RCWAResult

Run a full RCWA simulation and return an `RCWAResult`.

# Arguments

- `settings::RCWASettings`: Simulation parameters including the incoming wave,
  layer stack, and number of harmonics.
"""
function RCWA(s::RCWASettings)
    top_medium = s.stack.topMedium
    bottom_medium = s.stack.bottomMedium
    wavelength = s.incomingWave.wavelength

    # Prepare wave vectors
    wave_vectors = prepare_wave_vectors(
        s.incomingWave, top_medium, bottom_medium,
        s.stack.period, s.numberOfHarmonics
    )

    # Convolve layers
    top_c    = convolve(top_medium, s.numberOfHarmonics)
    bottom_c = convolve(bottom_medium, s.numberOfHarmonics)
    empty_c  = convolve(HomogeneousLayer(1.0), s.numberOfHarmonics)
    layers_c = [convolve(l, s.numberOfHarmonics) for l in s.stack.layers]

    # Compute eigenmodes
    top_modes = compute_modes(top_c, wave_vectors)
    bottom_modes = compute_modes(bottom_c, wave_vectors)
    empty_modes = compute_modes(empty_c, wave_vectors)
    layer_modes = [compute_modes(lc, wave_vectors) for lc in layers_c]

    # --- Global scattering matrix ---
    Sg = compute_global_scattering_matrix(
        top_modes, layer_modes, bottom_modes, empty_modes,
        wavelength, s.stack.thicknesses
    )

    return RCWAResult(Sg, wave_vectors, s)
end

"""Build the 2PQ incident source vector for the zeroth harmonic."""
function _source_vector(P::Int, Q::Int, polarization::Symbol)
    PQ = P * Q
    zeroth = Int(floor(P / 2)) + 1 + Int(floor(Q / 2)) * P
    c_inc = zeros(ComplexF64, 2PQ)
    if polarization === :x
        c_inc[zeroth] = 1.0
    elseif polarization === :y
        c_inc[zeroth + PQ] = 1.0
    else
        error("polarization must be :x or :y")
    end
    return c_inc
end

"""
    reflection_coefficients(result; polarization = :x) -> (r_x, r_y)

Complex reflected field amplitudes per diffraction order, returned as two
P x Q matrices (one per polarization component).
"""
function reflection_coefficients(
    result::RCWAResult;
    polarization::Symbol = :x,
)
    P, Q = result.input.numberOfHarmonics
    PQ = P * Q
    c_inc = _source_vector(P, Q, polarization)
    c_ref = result.scatteringMatrix.S11 * c_inc
    r_x = reshape(c_ref[1:PQ], P, Q)
    r_y = reshape(c_ref[PQ+1:2PQ], P, Q)
    return r_x, r_y
end

"""
    transmission_coefficients(result; polarization = :x) -> (t_x, t_y)

Complex transmitted field amplitudes per diffraction order, returned as two
P x Q matrices (one per polarization component).
"""
function transmission_coefficients(
    result::RCWAResult;
    polarization::Symbol = :x,
)
    P, Q = result.input.numberOfHarmonics
    PQ = P * Q
    c_inc = _source_vector(P, Q, polarization)
    c_trn = result.scatteringMatrix.S21 * c_inc
    t_x = reshape(c_trn[1:PQ], P, Q)
    t_y = reshape(c_trn[PQ+1:2PQ], P, Q)
    return t_x, t_y
end

"""
    diffraction_efficiencies(result; polarization = :x) -> (DE_ref, DE_trn)

Power per diffraction order normalized to the incident power, returned as two
P x Q real matrices. In the lossless case, `sum(DE_ref) + sum(DE_trn) ≈ 1`.

The power is computed from the z-component of the Poynting vector, which for a
plane wave with transverse fields (Ex, Ey) and wave vector (kx, ky, kz) is:

    Sz = Re(1/(kz* μ*)) x [(kz²+kx²)|Ex|² + 2 kx ky Re(Ex Ey*) + (ky²+kz²)|Ey|²]
"""
function diffraction_efficiencies(
    result::RCWAResult;
    polarization::Symbol = :x,
)
    P, Q = result.input.numberOfHarmonics
    PQ = P * Q
    top_mu = result.input.stack.topMedium.mu[1, 1]
    bottom_mu = result.input.stack.bottomMedium.mu[1, 1]

    r_x, r_y = reflection_coefficients(result; polarization)
    t_x, t_y = transmission_coefficients(result; polarization)

    kx = diag(result.waveVectors.waveVectorsX)
    ky = diag(result.waveVectors.waveVectorsY)
    K_top    = diag(result.waveVectors.waveVectorsTop)
    K_bottom = diag(result.waveVectors.waveVectorsBottom)

    zeroth = Int(floor(P / 2)) + 1 + Int(floor(Q / 2)) * P

    # Incident power (source is a single transverse polarization component).
    Ex_inc = polarization === :x ? 1.0 : 0.0
    Ey_inc = polarization === :y ? 1.0 : 0.0
    Sz_inc = _poynting_z(Ex_inc, Ey_inc, kx[zeroth], ky[zeroth],
                         -K_top[zeroth], top_mu)

    rx = vec(r_x); ry = vec(r_y)
    tx = vec(t_x); ty = vec(t_y)

    DE_ref = Vector{Float64}(undef, PQ)
    DE_trn = Vector{Float64}(undef, PQ)
    for j in 1:PQ
        # Reflected wave propagates in -z, so Sz is negative; negate for DE.
        DE_ref[j] = -_poynting_z(rx[j], ry[j], kx[j], ky[j],
                                 K_top[j], top_mu) / Sz_inc
        DE_trn[j] =  _poynting_z(tx[j], ty[j], kx[j], ky[j],
                                 K_bottom[j], bottom_mu) / Sz_inc
    end

    return reshape(DE_ref, P, Q), reshape(DE_trn, P, Q)
end

# z-component of the Poynting vector for a plane wave with transverse fields
# (Ex, Ey) and wave vector (kx, ky, kz) in a medium with permeability μ.
function _poynting_z(Ex, Ey, kx, ky, kz, μ)
    return real(((kz^2 + kx^2) * abs(Ex)^2
               + 2kx * ky * real(Ex * conj(Ey))
               + (ky^2 + kz^2) * abs(Ey)^2) / conj(kz * μ))
end