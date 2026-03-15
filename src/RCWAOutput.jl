"""
    struct RCWAOutput

Result of an RCWA simulation. Holds the global scattering matrix together with
all metadata needed to extract reflection/transmission coefficients and
diffraction efficiencies.

Use `reflection_coefficients`, `transmission_coefficients`, and
`diffraction_efficiencies` to query the result.

# Properties

- `input::RCWAInput`: Input that was used to produce this output.
- `waveVectors::PreparedWaveVectors`: Wave vectors for all diffraction orders.
- `modes::Vector{LayerModes}`: Eigenmodes of each layer (including top and
  bottom half-spaces as the first and last elements).
- `scatteringMatrix::ScatteringMatrix`: Global scattering matrix of the stack.
"""
struct RCWAOutput
    input::RCWAInput
    waveVectors::PreparedWaveVectors
    modes::Vector{LayerModes}
    scatteringMatrix::ScatteringMatrix
end

struct JonesVector
    x::ComplexF64
    y::ComplexF64
    
    function JonesVector(x::Number, y::Number)
        @assert (x != 0) || (y != 0)
        return new(x, y)
    end
end

# Build the 2PQ incident source vector for the zeroth harmonic.
function _source_vector(N::Int64, M::Int64, jones::JonesVector)
    P = 2N + 1
    Q = 2M + 1
    PQ = P * Q
    zeroth = N + 1 + M * P
    c_inc = zeros(ComplexF64, 2PQ)
    c_inc[zeroth] = jones.x
    c_inc[zeroth + PQ] = jones.y
    return c_inc
end

"""
    reflection_coefficients(output; polarization = :s) -> (r_x, r_y)

Complex reflected field amplitudes per diffraction order, returned as two
P x Q matrices (one per polarization component).

# Arguments

- `output::RCWAOutput`.
- `polarization`: Can be `:x`, `:y`, `:s` (TE), `:p` (TM), or an arbitrary 2-
  element Jones vector `[Ex, Ey]`.
"""
function reflection_coefficients(
    output::RCWAOutput;
    polarization = :s,
)
    N, M = output.input.order
    P = 2N + 1
    Q = 2M + 1
    PQ = P * Q
    jones = _resolve_polarization(polarization, output.input.source)
    c_inc = _source_vector(N, M, jones)
    c_ref = output.scatteringMatrix.S11 * c_inc
    r_x = reshape(c_ref[1:PQ], P, Q)
    r_y = reshape(c_ref[PQ+1:2PQ], P, Q)
    return r_x, r_y
end

"""
    transmission_coefficients(output; polarization = :s) -> (t_x, t_y)

Complex transmitted field amplitudes per diffraction order, returned as two
P x Q matrices (one per polarization component).

# Arguments

- `output::RCWAOutput`.
- `polarization`: Can be `:x`, `:y`, `:s` (TE), `:p` (TM), or an arbitrary 2-
  element Jones vector `[Ex, Ey]`.
"""
function transmission_coefficients(
    output::RCWAOutput;
    polarization = :s,
)
    N, M = output.input.order
    P = 2N + 1
    Q = 2M + 1
    PQ = P * Q
    jones = _resolve_polarization(polarization, output.input.source)
    c_inc = _source_vector(N, M, jones)
    c_trn = output.scatteringMatrix.S21 * c_inc
    t_x = reshape(c_trn[1:PQ], P, Q)
    t_y = reshape(c_trn[PQ+1:2PQ], P, Q)
    return t_x, t_y
end

# Resolve a polarization keyword to a Jones vector [Ex, Ey].
function _resolve_polarization(pol::Symbol, source::Source)
    if pol === :x
        return JonesVector(1, 0)
    elseif pol === :y
        return JonesVector(0, 1)
    elseif pol === :s
        ϕ = source.azimuthalAngle
        return JonesVector(-sin(ϕ), cos(ϕ))
    elseif pol === :p
        ϕ = source.azimuthalAngle
        θ = source.elevationAngle
        return JonesVector(cos(θ) * cos(ϕ), cos(θ) * sin(ϕ))
    else
        error("polarization must be :x, :y, :s, :p, or a 2-element Jones vector")
    end
end

function _resolve_polarization(pol::AbstractVector{<:Number}, ::Source)
    @assert length(pol) == 2
    return JonesVector(pol[1], pol[2])
end

_resolve_polarization(pol::JonesVector, ::Source) = pol

"""
    diffraction_efficiencies(output; polarization = :s) -> (DE_ref, DE_trn)

Power per diffraction order normalized to the incident power, returned as two
P x Q real matrices. The power is computed from the z-component of the Poynting
vector.

# Arguments

- `output::RCWAOutput`.
- `polarization`: Can be `:x`, `:y`, `:s` (TE), `:p` (TM), or an arbitrary 2-
  element Jones vector `[Ex, Ey]`.
"""
function diffraction_efficiencies(
    output::RCWAOutput;
    polarization = :s,
)
    N, M = output.input.order
    P = 2N + 1
    Q = 2M + 1
    PQ = P * Q
    top_mu = first(first(output.input.stack.layers).mu)
    bottom_mu = first(last(output.input.stack.layers).mu)

    jones = _resolve_polarization(polarization, output.input.source)
    r_x, r_y = reflection_coefficients(output; polarization)
    t_x, t_y = transmission_coefficients(output; polarization)

    kx        = diag(output.waveVectors.waveVectorsX)
    ky        = diag(output.waveVectors.waveVectorsY)
    kz_top    = diag(output.waveVectors.waveVectorsZReflection)
    kz_bottom = diag(output.waveVectors.waveVectorsZTransmission)

    zeroth = N + 1 + M * P

    # Incident power from the Jones vector.
    Sz_inc = _poynting_z(jones.x, jones.y, kx[zeroth], ky[zeroth],
                         -kz_top[zeroth], top_mu)

    rx = vec(r_x); ry = vec(r_y)
    tx = vec(t_x); ty = vec(t_y)

    DE_ref = Vector{Float64}(undef, PQ)
    DE_trn = Vector{Float64}(undef, PQ)
    for j in 1:PQ
        # Reflected wave propagates in -z, so Sz is negative; negate for DE.
        DE_ref[j] = -_poynting_z(rx[j], ry[j], kx[j], ky[j],
                                 kz_top[j], top_mu) / Sz_inc
        DE_trn[j] =  _poynting_z(tx[j], ty[j], kx[j], ky[j],
                                 kz_bottom[j], bottom_mu) / Sz_inc
    end

    return reshape(DE_ref, P, Q), reshape(DE_trn, P, Q)
end

# z-component of the Poynting vector for a plane wave with transverse fields
# (Ex, Ey) and wave vector (kx, ky, kz) in a medium with permeability μ.
function _poynting_z(Ex, Ey, kx, ky, kz, μ)
    if kz == 0 # Grazing mode; no power in z.
        return 0.0
    end
    return real(((kz^2 + kx^2) * abs(Ex)^2
               + 2kx * ky * real(Ex * conj(Ey))
               + (ky^2 + kz^2) * abs(Ey)^2) / conj(kz * μ))
end