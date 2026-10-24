"""
    struct Source

Describes incoming wave in a vacuum.

# Properties

- `azimuthalAngle::Float64`: Also called ϕ. Some people call it θ.
- `elevationAngle::Float64`: Also called θ. Some people call it ϕ.
- `wavelength::Float64`: Wavelength in whatever units you like so long as you're
  being consistent.
"""
struct Source
    azimuthalAngle::Float64 # ϕ
    elevationAngle::Float64 # θ
    wavelength::Float64 # λ

    function Source(
        azimuthal_angle::Number,
        elevation_angle::Number,
        wavelength::Number
    )
        @assert wavelength > 0
        return new(azimuthal_angle, elevation_angle, wavelength)
    end
end

function Source(wavelength::Number)
    return Source(0.0, 0.0, wavelength)
end

"""
    struct PreparedWaveVectors

Wave vectors presented in a shape convenient for subsequent simulation steps.

# Properties

- `waveVectorsX::Diagonal{ComplexF64}`: Diagonal matrix of wave vectors in X.
- `waveVectorsY::Diagonal{ComplexF64}`: Diagonal matrix of wave vectors in Y.
- `waveVectorsZReflection::Diagonal{ComplexF64}`: Diagonal matrix of the z-
  components of the reflected wave vectors. These travel (or decay) in the -z
  direction.
- `waveVectorsZTransmission::Diagonal{ComplexF64}`: Diagonal matrix of the z-
  components of the transmitted wave vectors. These travel (or decay) in the +z
  direction.

All wave vectors are normalized by the free-space wave number 2π/λ. Harmonics
are ordered with the X order varying fastest, and order `m` has in-plane wave
vector `k_x + m λ/Λ`.
"""
struct PreparedWaveVectors
    waveVectorsX::Diagonal{ComplexF64}
    waveVectorsY::Diagonal{ComplexF64}
    waveVectorsZReflection::Diagonal{ComplexF64}
    waveVectorsZTransmission::Diagonal{ComplexF64}
end

"""
    function prepare_wave_vectors(
        source,
        top_medium,
        bottom_medium,
        period,
        number_of_harmonics
    )

Prepare wave vectors in a format convenient for subsequent RCWA simulation
steps.

# Arguments

- `source::Source`: Incoming wave as described in a vacuum.
- `top_medium::Layer`: Homogeneous reflection region.
- `bottom_medium::Layer`: Homogeneous transmission region.
- `period::Tuple{<:Real, <:Real}`: Periodicity, or equivalently, size of your
  unit cells.
- `number_of_harmonics::Tuple{<:Integer, <:Integer}`: Number of harmonics to be
  used in your RCWA simulation.
"""
function prepare_wave_vectors(
    source::Source,
    top_medium::Layer,
    bottom_medium::Layer,
    period::Tuple{<:Real, <:Real},
    number_of_harmonics::Tuple{<:Integer, <:Integer}
)
    @assert _is_homogeneous(top_medium)
    @assert _is_homogeneous(bottom_medium)
    @assert (period[1] > 0) && (period[2] > 0)
    @assert (number_of_harmonics[1] > 0) && (number_of_harmonics[2] > 0)
    
    ϕ = source.azimuthalAngle
    θ = source.elevationAngle
    λ = source.wavelength
  
    P, Q = number_of_harmonics
    eps_top = first(top_medium.eps)
    mu_top = first(top_medium.mu)
    n_top = sqrt(eps_top * mu_top)
    eps_bottom = first(bottom_medium.eps)
    mu_bottom = first(bottom_medium.mu)

    k_x = n_top * sin(θ) * cos(ϕ)
    k_y = n_top * sin(θ) * sin(ϕ)

    # Diffraction order m has in-plane wave vector k_x + m λ/Λ (grating
    # equation, normalized by 2π/λ).
    P_range = -(P ÷ 2):(P ÷ 2)
    Q_range = -(Q ÷ 2):(Q ÷ 2)

    K_x = repeat(k_x .+ (λ / period[1]) .* P_range; outer = Q)
    K_y = repeat(k_y .+ (λ / period[2]) .* Q_range; inner = P)

    # Reflected waves travel in the -z direction, transmitted waves in +z.
    K_top    = .-_forward_kz.(eps_top    * mu_top    .- K_x.^2 .- K_y.^2)
    K_bottom =   _forward_kz.(eps_bottom * mu_bottom .- K_x.^2 .- K_y.^2)

    return PreparedWaveVectors(
        Diagonal(K_x), Diagonal(K_y), Diagonal(K_top), Diagonal(K_bottom)
    )
end

"""
    _forward_kz(kz2)

Return the square root `kz` of `kz2` that describes a wave travelling or
decaying in the +z direction, i.e. `exp(i kz z)` is bounded for z > 0. This
means `imag(kz) > 0`, or `imag(kz) == 0` and `real(kz) >= 0`.
"""
function _forward_kz(kz2::Number)
    kz = sqrt(complex(kz2))
    # `sqrt` has its branch cut along the negative real axis, so for a lossless
    # evanescent wave the sign of a zero imaginary part would decide the branch.
    # Such signed zeros arise naturally, e.g. (-0.5 + 0.0im)^2 == 0.25 - 0.0im.
    return imag(kz) < 0 ? -kz : kz
end