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
- `waveVectorsTop::Diagonal{ComplexF64}`: Diagonal matrix of wave vectors in
  reflection region.
- `waveVectorsBottom::Diagonal{ComplexF64}`: Diagonal matrix of wave vectors in
  transmission region.
"""
struct PreparedWaveVectors
    waveVectorsX::Diagonal{ComplexF64}
    waveVectorsY::Diagonal{ComplexF64}
    waveVectorsTop::Diagonal{ComplexF64}
    waveVectorsBottom::Diagonal{ComplexF64}
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
    @assert is_homogeneous(top_medium)
    @assert is_homogeneous(bottom_medium)
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

    P_range = Vector(-floor(P / 2):floor(P / 2))
    Q_range = Vector(-floor(Q / 2):floor(Q / 2))

    K_x = k_x .- (λ / period[1]) * P_range
    K_y = k_y .- (λ / period[2]) * Q_range

    K_x = Diagonal(repeat(K_x; outer = Q))
    K_y = Diagonal(repeat(K_y; inner = P))

    # Conjugate is needed to deal with negative sign convention.
    K_top    = -conj.(sqrt.(conj.(eps_top    * mu_top    * I(P * Q) - K_x^2 - K_y^2)))
    K_bottom =        sqrt.(conj.(eps_bottom * mu_bottom * I(P * Q) - K_x^2 - K_y^2))

    return PreparedWaveVectors(K_x, K_y, K_top, K_bottom)
end