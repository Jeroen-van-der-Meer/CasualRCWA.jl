struct LayerModes
    eigenvalues::Vector{ComplexF64}
    electricModes::AbstractMatrix{ComplexF64} # DiagonalBlocks if homogeneous
    magneticModes::AbstractMatrix{ComplexF64} # DiagonalBlocks if homogeneous

    function LayerModes(
        eigenvalues::AbstractVector{<:Number},
        electric_modes::AbstractMatrix{<:Number},
        magnetic_modes::AbstractMatrix{<:Number}
    )
        @assert length(eigenvalues) == size(electric_modes, 1) ==
            size(electric_modes, 2) == size(magnetic_modes, 1) ==
            size(magnetic_modes, 2)
        return new(eigenvalues, electric_modes, magnetic_modes)
    end
end

const RAYLEIGH_TOL = 1e-12 # FIXME: Figure out if these constants are sane
const RAYLEIGH_PERTURB = 1e-11

function compute_modes(
    layer::ConvolvedLayer,
    wave_data::PreparedWaveVectors
)
    E = layer.conv_eps
    M = layer.conv_mu
    Kx = wave_data.waveVectorsX
    Ky = wave_data.waveVectorsY
    if _is_homogeneous(layer)
        # For homogeneous layers (E and M are scalar multiples of identity), the
        # eigenvalue problem has degenerate eigenvalues, and eigen() returns
        # arbitrary eigenvectors. This causes problems when comparing modes
        # between layers of the same material. Use the analytical solution
        # instead.
        return compute_modes_homogeneous(first(E), first(M), Kx, Ky)
    else
        return compute_modes(E, M, Kx, Ky)
    end
end

function compute_modes(
    E::AbstractMatrix{ComplexF64},
    M::AbstractMatrix{ComplexF64},
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64}
)
    # Precompute some terms.
    EKx = E \ Kx
    EKy = E \ Ky
    MKx = M \ Kx
    MKy = M \ Ky

    # The electric and magnetic field components in our layer satisfy the
    # coupled wave equations dH/dz = Q E and dE/dz = P E.
    P = [
        Kx*EKy      -Kx*EKx+M;
        Ky*EKy-M    -Ky*EKx
    ]
    Q = [
        Kx*MKy      -Kx*MKx+E;
        Ky*MKy-E    -Ky*MKx
    ]
    # Alternatively, we may express the E-field in terms of the more familiar
    # equation d^2E/dz^2 = PQ E.
    Ω2 = P * Q

    # The solution to this equation is expressible in terms of the eigenmodes
    # of Ω, so we may precompute these.
    eigenvalues2, E_modes = _eigen_blockwise!(Ω2)

    _stabilize_eigenvectors!(eigenvalues2, E_modes)
    
    # Rayleigh anomaly occurs when you have grazing harmonics with zero
    # eigenvalues. These cause singularities in the scattering matrices. We
    # perturb the squared eigenvalues by adding an infinitesimal loss, which
    # adds a positive imaginary part to ε and hence a negative one to λ².
    Rayleigh_anomalies = abs.(eigenvalues2) .< RAYLEIGH_TOL
    eigenvalues2[Rayleigh_anomalies] .-= im * RAYLEIGH_PERTURB

    # The principal square root has a nonnegative real part, so exp(-λz) never
    # grows in the +z direction. For propagating modes (λ purely imaginary) the
    # sign is arbitrary, but this does not affect the scattering matrix of a
    # layer, which is independent of how its internal modes are labeled.
    eigenvalues = sqrt.(eigenvalues2)

    # The magnetic fields adhere to a similar wave equation. Its eigenvalues are
    # the same as those of the E-field, and its eigenmodes are directly
    # inferred from those of the E-field.
    M_modes = Q * E_modes * Diagonal(1 ./ eigenvalues)
    return LayerModes(eigenvalues, E_modes, M_modes)
end

"""
Compute eigenmodes analytically for a homogeneous layer.

In a homogeneous medium, all spatial harmonics decouple, so the eigenvectors are
simply the identity matrix. The eigenvalues are determined by the dispersion
relation: λ_n = -i kz_n with kz_n = sqrt(εμ - kx_n² - ky_n²).

Unlike for patterned layers, the branch of the square root matters here: the
modes of the half-spaces and of the free-space gap medium define which waves
count as incoming and outgoing in the scattering matrices. The forward mode
exp(-λ z) = exp(i kz z) must travel or decay in the +z direction.
"""
function compute_modes_homogeneous(
    ε::ComplexF64,
    μ::ComplexF64,
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64}
)
    PQ = size(Kx, 1)

    kx = diag(Kx)
    ky = diag(Ky)
    kz2 = ε * μ .- kx.^2 .- ky.^2

    # Rayleigh anomaly occurs when you have grazing harmonics with zero
    # eigenvalues. These cause singularities in the scattering matrices. We
    # perturb the permittivity by adding an infinitesimal loss.
    Rayleigh_anomalies = abs.(kz2) .< RAYLEIGH_TOL
    ε_pert = fill(ε, PQ)
    ε_pert[Rayleigh_anomalies] .+= im * RAYLEIGH_PERTURB
    kz2[Rayleigh_anomalies] .+= im * RAYLEIGH_PERTURB * μ

    # Each spatial harmonic n has eigenvalue -i kz_n, repeated for both
    # polarizations.
    eigenvalues = -im .* _forward_kz.(kz2)
    eigenvalues = vcat(eigenvalues, eigenvalues)

    # Eigenvectors are the identity (harmonics decouple in homogeneous media).
    E_modes = DiagonalBlocks(I, PQ)

    # Magnetic modes: M_modes = Q * E_modes * diag(1/eigenvalues)
    # Since E_modes = I, this simplifies to Q * diag(1/eigenvalues).
    Q = DiagonalBlocks(
        kx .* ky ./ μ, ε_pert .- kx.^2 ./ μ,
        ky.^2 ./ μ .- ε_pert, -kx .* ky ./ μ
    )
    M_modes = Q * Diagonal(1 ./ eigenvalues)

    return LayerModes(eigenvalues, E_modes, M_modes)
end

"""
Eigendecomposition of Ω² that exploits decoupled polarizations.

When the off-diagonal blocks of Ω² vanish, the X and Y components of the
electric field decouple, and the two diagonal blocks may be diagonalized
separately.
"""
function _eigen_blockwise!(Ω2::Matrix{ComplexF64})
    n = size(Ω2, 1) ÷ 2
    top, bottom = 1:n, (n + 1):2n
    if iszero(view(Ω2, top, bottom)) && iszero(view(Ω2, bottom, top))
        F1 = eigen!(Ω2[top, top])
        F2 = eigen!(Ω2[bottom, bottom])
        vectors = zeros(ComplexF64, 2n, 2n)
        vectors[top, top] = F1.vectors
        vectors[bottom, bottom] = F2.vectors
        return vcat(F1.values, F2.values), vectors
    else
        F = eigen!(Ω2)
        return F.values, F.vectors
    end
end

"""
Stabilize eigenvectors of nearly-degenerate eigenvalue clusters.

When `eigen()` encounters degenerate or nearly-degenerate eigenvalues, it returns
arbitrary eigenvectors within each degenerate subspace. This causes numerical
issues downstream (ill-conditioned W⁻¹ in S-matrix boundary matching).

This function groups nearly-degenerate eigenvalues into clusters and, within each
cluster, applies orthogonal Procrustes alignment to rotate the eigenvectors so
they best align with the identity matrix (the free-space eigenvector basis).
"""
function _stabilize_eigenvectors!(
    eigenvalues2::Vector{ComplexF64},
    eigenvectors::Matrix{ComplexF64};
    tol::Float64 = 1e-6
)
    n = length(eigenvalues2)

    # Sort by real part (breaking ties by imaginary part) for deterministic
    # ordering across platforms and Julia versions.
    perm = sortperm(eigenvalues2, by = λ -> (real(λ), imag(λ)))
    eigenvalues2 .= eigenvalues2[perm]
    eigenvectors .= eigenvectors[:, perm]

    # Identify clusters of nearly-degenerate eigenvalues and align each.
    i = 1
    while i <= n
        j = i
        while j < n && abs(eigenvalues2[j + 1] - eigenvalues2[i]) < tol * (1 + abs(eigenvalues2[i]))
            j += 1
        end
        if j > i
            _align_cluster!(eigenvectors, i:j)
        end
        i = j + 1
    end
    return nothing
end

# Orthogonal Procrustes: rotate eigenvectors in `cols` so they best align with
# the corresponding columns of the identity matrix.
function _align_cluster!(W::Matrix{ComplexF64}, cols::UnitRange{Int})
    W_c = W[:, cols]
    # The "target" is the identity columns at indices `cols`.  The projection
    # of these onto the cluster subspace is simply W_c[cols, :].
    M = W_c[cols, :]
    U, _, V = svd(M)
    # Best-aligning unitary rotation: R = V * U'
    W[:, cols] = W_c * (V * U')
end

function number_of_modes(layer_modes::LayerModes)
    return length(layer_modes.eigenvalues)
end
