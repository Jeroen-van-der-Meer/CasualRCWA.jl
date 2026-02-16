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

function compute_modes(
    layer::ConvolvedLayer,
    wave_data::PreparedWaveVectors
)
    E = layer.conv_eps
    M = layer.conv_mu
    Kx = wave_data.waveVectorsX
    Ky = wave_data.waveVectorsY

    return compute_modes(E, M, Kx, Ky)
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
    eigenvalues, E_modes = eigen(Ω2)
    eigenvalues .= sqrt.(eigenvalues)

    # The magnetic fields adhere to a similar wave equation. Its eigenvalues are
    # the same as those of the E-field, and its eigenmodes are directly
    # inferred from those of the E-field.
    M_modes = Q * E_modes * Diagonal(1 ./ eigenvalues)
    return LayerModes(eigenvalues, E_modes, M_modes)
end

function number_of_modes(layer_modes::LayerModes)
    return length(layer_modes.eigenvalues)
end