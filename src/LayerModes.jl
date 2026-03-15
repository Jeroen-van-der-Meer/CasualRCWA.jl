struct LayerModes
    eigenvalues::Vector{ComplexF64}
    electricModes::Matrix{ComplexF64}
    magneticModes::Matrix{ComplexF64}

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
    if is_homogeneous(layer)
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
    eigenvalues2, E_modes = eigen(Ω2)

    _stabilize_eigenvectors!(eigenvalues2, E_modes)
    
    # Rayleigh anomaly occurs when you have grazing harmonics with zero
    # eigenvalues. These cause singularities in the scattering matrices. We
    # perturb the squared eigenvalues by adding an infinitesimal loss.
    Rayleigh_anomalies = abs.(eigenvalues2) .< RAYLEIGH_TOL
    eigenvalues2[Rayleigh_anomalies] .+= im * RAYLEIGH_PERTURB

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
relation: λ_n = sqrt(kx_n² + ky_n² - εμ).
"""
function compute_modes_homogeneous(
    ε::ComplexF64,
    μ::ComplexF64,
    Kx::AbstractMatrix{ComplexF64},
    Ky::AbstractMatrix{ComplexF64}
)
    PQ = size(Kx, 1)

    # Each spatial harmonic n has eigenvalue sqrt(kx_n² + ky_n² - εμ),
    # repeated for both polarizations.
    kx = diag(Kx)
    ky = diag(Ky)
    eigenvalues2 = Complex.(kx.^2 .+ ky.^2 .- ε * μ)

    # Rayleigh anomaly occurs when you have grazing harmonics with zero
    # eigenvalues. These cause singularities in the scattering matrices. We
    # perturb the squared eigenvalues by adding an infinitesimal loss.
    Rayleigh_anomalies = abs.(eigenvalues2) .< RAYLEIGH_TOL
    ε_pert = fill(ε, PQ)
    ε_pert[Rayleigh_anomalies] .-= im * RAYLEIGH_PERTURB
    eigenvalues2[Rayleigh_anomalies] .+= im * RAYLEIGH_PERTURB * μ

    eigenvalues = sqrt.(eigenvalues2)
    eigenvalues = vcat(eigenvalues, eigenvalues)

    # Eigenvectors are the identity (harmonics decouple in homogeneous media).
    E_modes = Matrix{ComplexF64}(I, 2PQ, 2PQ)

    # Magnetic modes: M_modes = Q * E_modes * diag(1/eigenvalues)
    # Since E_modes = I, this simplifies to Q * diag(1/eigenvalues).
    MKx = Kx / μ
    MKy = Ky / μ
    Q = Matrix{ComplexF64}([
        Kx*MKy                   -Kx*MKx+Diagonal(ε_pert);
        Ky*MKy-Diagonal(ε_pert)  -Ky*MKx
    ])
    M_modes = Q * Diagonal(1 ./ eigenvalues)

    return LayerModes(eigenvalues, E_modes, M_modes)
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
