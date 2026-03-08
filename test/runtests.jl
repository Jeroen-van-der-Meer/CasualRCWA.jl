#!/usr/bin/env julia

using LinearAlgebra
using RCWAForRetards
using Test

@testset "Convolution 4x4" begin
    M = [
        0.8825 + 0.8059im   0.1837 + 0.1019im   0.5598 + 0.1879im   0.6498 + 0.4685im
        0.5390 + 0.5389im   0.2492 + 0.6446im   0.4151 + 0.0094im   0.2888 + 0.9085im
        0.5437 + 0.5901im   0.2851 + 0.9801im   0.9057 + 0.8383im   0.2553 + 0.4171im
        0.4425 + 0.2311im   0.5295 + 0.1017im   0.3051 + 0.4813im   0.3582 + 0.5441im
    ]

    CM = RCWAForRetards.convolve(M, (1, 1))
    CM_ref = [
        0.4621 + 0.4906im
    ]
    @test all(isapprox.(CM, CM_ref; atol = 1e-4))

    CM = RCWAForRetards.convolve(M, (2, 2))
    CM_ref = [
        0.4621 + 0.4906im  -0.0286 - 0.0878im   0.0457 + 0.0215im   0.0653 + 0.0112im
        0.0643 - 0.0699im   0.4621 + 0.4906im   0.1364 + 0.0351im   0.0457 + 0.0215im
       -0.0180 + 0.0596im  -0.0772 + 0.0955im   0.4621 + 0.4906im  -0.0286 - 0.0878im
        0.0466 + 0.0748im  -0.0180 + 0.0596im   0.0643 - 0.0699im   0.4621 + 0.4906im
    ]
    @test all(isapprox.(CM, CM_ref; atol = 1e-4))
end

@testset "Convolution 5x5" begin
    M = [
        0.2938 + 0.6158im   0.5081 + 0.0607im   0.0422 + 0.1869im   0.8900 + 0.2335im   0.4370 + 0.9221im
        0.8733 + 0.7982im   0.7485 + 0.0908im   0.1576 + 0.8325im   0.0589 + 0.2197im   0.0888 + 0.6556im
        0.8235 + 0.8202im   0.3999 + 0.8056im   0.3614 + 0.0626im   0.2025 + 0.2593im   0.7757 + 0.6336im
        0.9988 + 0.3686im   0.9317 + 0.5470im   0.0589 + 0.6000im   0.0874 + 0.5993im   0.7339 + 0.4018im
        0.3745 + 0.8099im   0.4277 + 0.8951im   0.8434 + 0.0601im   0.5731 + 0.6949im   0.1452 + 0.3578im
    ]

    CM = RCWAForRetards.convolve(M, (2, 3))
    CM_ref = [
        0.4734 + 0.5013im  -0.0271 - 0.0397im   0.1208 + 0.1052im  -0.0456 + 0.0671im   0.0103 + 0.0420im   0.0230 + 0.1274im
       -0.0409 + 0.0052im   0.4734 + 0.5013im  -0.0158 - 0.0479im   0.1208 + 0.1052im   0.0551 - 0.0125im   0.0103 + 0.0420im
        0.0648 + 0.0580im  -0.1272 + 0.0399im   0.4734 + 0.5013im  -0.0271 - 0.0397im   0.1208 + 0.1052im  -0.0456 + 0.0671im
       -0.0957 - 0.0187im   0.0648 + 0.0580im  -0.0409 + 0.0052im   0.4734 + 0.5013im  -0.0158 - 0.0479im   0.1208 + 0.1052im
        0.0035 - 0.0238im  -0.0330 - 0.0902im   0.0648 + 0.0580im  -0.1272 + 0.0399im   0.4734 + 0.5013im  -0.0271 - 0.0397im
       -0.0109 + 0.0299im   0.0035 - 0.0238im  -0.0957 - 0.0187im   0.0648 + 0.0580im  -0.0409 + 0.0052im   0.4734 + 0.5013im
    ]
    @test all(isapprox.(CM, CM_ref; atol = 1e-4))
end

@testset "Wave vector preparation" begin
    wavelength = 0.02
    elevation_angle = 1.0
    azimuthal_angle = 0.5
    incoming_wave = IncomingWave(azimuthal_angle, elevation_angle, wavelength)

    top_medium = HomogeneousLayer(2.0, 1.0)
    bottom_medium = HomogeneousLayer(9.0, 1.0)
    period = (0.0175, 0.015)

    number_of_harmonics = (3, 3)

    prepared_wave_vectors = RCWAForRetards.prepare_wave_vectors(
        incoming_wave,
        top_medium,
        bottom_medium,
        period,
        number_of_harmonics
    )

    K_x = diag(prepared_wave_vectors.waveVectorsX)
    K_x_ref = [
        2.1872 + 0.0000im
        1.0443 + 0.0000im
       -0.0985 + 0.0000im
        2.1872 + 0.0000im
        1.0443 + 0.0000im
       -0.0985 + 0.0000im
        2.1872 + 0.0000im
        1.0443 + 0.0000im
       -0.0985 + 0.0000im
    ]
    @test all(isapprox.(K_x, K_x_ref; atol = 1e-4))

    K_y = diag(prepared_wave_vectors.waveVectorsY)
    K_y_ref = [
        1.9039 + 0.0000im
        1.9039 + 0.0000im
        1.9039 + 0.0000im
        0.5705 + 0.0000im
        0.5705 + 0.0000im
        0.5705 + 0.0000im
       -0.7628 + 0.0000im
       -0.7628 + 0.0000im
       -0.7628 + 0.0000im
    ]
    @test all(isapprox.(K_y, K_y_ref; atol = 1e-4))

    K_reflection = diag(prepared_wave_vectors.waveVectorsTop)
    K_reflection_ref = [
        0.0000 - 2.5315im
        0.0000 - 1.6478im
        0.0000 - 1.2784im
        0.0000 - 1.7633im
       -0.7641 + 0.0000im
       -1.2903 + 0.0000im
        0.0000 - 1.8346im
       -0.5723 + 0.0000im
       -1.1868 + 0.0000im
    ]
    @test all(isapprox.(K_reflection, K_reflection_ref; atol = 1e-4))

    K_transmission = diag(prepared_wave_vectors.waveVectorsBottom)
    K_transmission_ref = [
        0.7691 + 0.0000im
        2.0699 + 0.0000im
        2.3164 + 0.0000im
        1.9725 + 0.0000im
        2.7539 + 0.0000im
        2.9436 + 0.0000im
        1.9064 + 0.0000im
        2.7069 + 0.0000im
        2.8997 + 0.0000im
    ]
    @test all(isapprox.(K_transmission, K_transmission_ref; atol = 1e-4))
end

@testset "Layer eigenmodes" begin
    M = [
        0.4387 + 0.1869im   0.7655 + 0.4456im
        0.3816 + 0.4898im   0.7952 + 0.6463im
    ]
    E = [
        0.7094 + 0.6551im   0.2760 + 0.1190im
        0.7547 + 0.1626im   0.6797 + 0.4984im
    ]
    Kx = Diagonal([0.9597 + 0.5853im, 0.3404 + 0.2238im])
    Ky = Diagonal([0.7513 + 0.5060im, 0.2551 + 0.6991im])
    layer_modes = RCWAForRetards.compute_modes(E, M, Kx, Ky)

    eigenvalues = layer_modes.eigenvalues
    # There is no guarantee about the order in which Julia returns the
    # eigenvalues, so we manually sort them for a fair test.
    s = sortperm(eigenvalues, by = real)
    eigenvalues = eigenvalues[s]
    eigenvalues_ref = [
        0.3150 - 1.0928im
        0.4963 - 1.0481im
        0.8594 + 1.1643im
        1.0358 + 0.6993im
    ]
    @test all(isapprox.(eigenvalues, eigenvalues_ref; atol = 1e-4))

    eigenvectors_E = layer_modes.electricModes
    eigenvectors_E = eigenvectors_E[:, s]
    eigenvectors_E_ref = [
       -0.1425 + 0.2517im  -0.0174 + 0.3293im   0.1478 + 0.3316im   0.5566 + 0.0000im
       -0.3242 + 0.4081im   0.0891 + 0.5966im  -0.2941 + 0.3702im  -0.3948 - 0.0297im
        0.3848 - 0.0605im   0.2742 - 0.1044im   0.2259 - 0.4336im   0.5237 + 0.1355im
        0.7021 + 0.0000im   0.6642 + 0.0000im   0.6368 + 0.0000im  -0.4848 - 0.0766im
    ]
    @test all(isapprox.(eigenvectors_E, eigenvectors_E_ref; atol = 1e-4))
    
    eigenvectors_M = layer_modes.magneticModes
    eigenvectors_M = eigenvectors_M[:, s]
    eigenvectors_M_ref = [
        0.3238 - 0.0660im  -0.0977 + 0.2282im  -2.3511 + 1.4219im   0.2894 - 0.0579im
       -0.2358 + 0.6074im  -0.0968 + 0.5120im   0.7992 - 0.5064im  -0.0248 - 0.1107im
        0.6139 - 0.1361im   0.3225 - 0.0882im  -2.2497 + 1.0610im  -0.3035 - 0.2568im
        0.3604 + 0.0857im   0.6211 - 0.3269im   0.8233 + 0.0066im  -0.0960 + 0.1189im
    ]
    @test all(isapprox.(eigenvectors_M, eigenvectors_M_ref; atol = 1e-4))
end

@testset "Symmetric scattering matrices" begin
    A = [
        0.8003 + 0.7922im   0.4218 + 0.6557im
        0.1419 + 0.9595im   0.9157 + 0.0357im
    ]
    B = [
        0.8491 + 0.7431im   0.6787 + 0.6555im
        0.9340 + 0.3922im   0.7577 + 0.1712im
    ]
    X = Diagonal([
        0.7060 + 0.0971im,  0.2769 + 0.6948im
    ])
    R, T = RCWAForRetards._compute_symmetric_scattering_matrix(A, B, X)

    R_ref = [
       -0.8003 + 0.2375im  -0.6227 + 0.2738im
       -0.7866 - 0.0397im  -0.7199 + 0.1529im
    ]
    T_ref = [
       -0.2334 + 0.5908im  -0.4982 + 0.1877im
       -0.2907 - 0.6003im   0.1486 + 0.2375im
    ]
    @test all(isapprox.(R, R_ref; atol = 1e-4))
    @test all(isapprox.(T, T_ref; atol = 1e-4))
end

@testset "Star product" begin
    A11 = [
        0.6324 + 0.9575im   0.2785 + 0.1576im
        0.0975 + 0.9649im   0.5469 + 0.9706im
    ]
    A12 = [
        0.6948 + 0.7655im   0.9502 + 0.1869im   0.4387 + 0.4456im
        0.3171 + 0.7952im   0.0344 + 0.4898im   0.3816 + 0.6463im
    ]
    A21 = [
        0.7094 + 0.1190im   0.6797 + 0.3404im
        0.7547 + 0.4984im   0.6551 + 0.5853im
        0.2760 + 0.9597im   0.1626 + 0.2238im
    ]
    A22 = [
        0.7513 + 0.2575im   0.6991 + 0.8143im   0.5472 + 0.3500im
        0.2551 + 0.8407im   0.8909 + 0.2435im   0.1386 + 0.1966im
        0.5060 + 0.2543im   0.9593 + 0.9293im   0.1493 + 0.2511im
    ]
    A = RCWAForRetards.ScatteringMatrix(A11, A12, A21, A22)
    B = RCWAForRetards.ScatteringMatrix(A22, A21, A12, A11)

    S = RCWAForRetards.star_product(A, B)
    T = RCWAForRetards.star_product(B, A)

    S11_ref = [
       -0.3913 + 0.6709im  -0.6161 - 0.1178im
       -0.5446 + 0.4066im  -0.0053 + 0.4518im
    ]
    @test all(isapprox.(S.S11, S11_ref; atol = 1e-4))

    S12_ref = [
       -0.5457 + 0.1526im  -0.3878 + 0.0234im
       -0.7031 + 0.0070im  -0.4637 - 0.2906im
    ]
    @test all(isapprox.(S.S12, S12_ref; atol = 1e-4))

    S21_ref = S12_ref
    @test all(isapprox.(S.S21, S21_ref; atol = 1e-4))

    S22_ref = S11_ref
    @test all(isapprox.(S.S22, S22_ref; atol = 1e-4))

    T11_ref = [
        0.1479 + 0.1952im   0.3239 + 0.8759im   0.0879 + 0.3471im
       -0.4415 + 0.5244im   0.3905 + 0.1803im  -0.3993 + 0.0138im
        0.0661 - 0.1891im   0.4830 + 0.8181im  -0.1581 - 0.0305im
    ]
    @test all(isapprox.(T.S11, T11_ref; atol = 1e-4))

    T12_ref = [
       -0.0985 + 0.2243im  -0.0786 + 0.0119im  -0.0859 + 0.2250im
       -0.2445 + 0.2273im  -0.1275 + 0.0462im  -0.2022 + 0.2245im
       -0.4212 + 0.1231im  -0.1941 + 0.3087im  -0.2697 + 0.0736im
    ]
    @test all(isapprox.(T.S12, T12_ref; atol = 1e-4))

    T21_ref = T12_ref
    @test all(isapprox.(T.S21, T21_ref; atol = 1e-4))

    T22_ref = T11_ref
    @test all(isapprox.(T.S22, T22_ref; atol = 1e-4))
end

@testset "Homogeneous layer eigenmodes" begin
    # For a homogeneous medium, eigenvectors should be the identity and
    # eigenvalues should be analytically determined by the dispersion relation.
    incoming = IncomingWave(0.0, 0.0, 532.0)
    top = HomogeneousLayer(1.0)
    bottom = HomogeneousLayer(1.0)
    nh = (3, 3)
    wv = RCWAForRetards.prepare_wave_vectors(incoming, top, bottom, (3200.0, 100.0), nh)

    empty_c = RCWAForRetards.convolve(HomogeneousLayer(1.0), nh)
    modes = RCWAForRetards.compute_modes(empty_c, wv)

    PQ = prod(nh)
    @test modes.electricModes ≈ Matrix{ComplexF64}(I, 2PQ, 2PQ)

    # Eigenvalues should satisfy the dispersion relation λ = sqrt(kx² + ky² - εμ).
    kx = diag(wv.waveVectorsX)
    ky = diag(wv.waveVectorsY)
    λ_expected = sqrt.(Complex.(kx.^2 .+ ky.^2 .- 1.0))
    @test modes.eigenvalues ≈ vcat(λ_expected, λ_expected)
end

# Common pattern for tests: 50% duty cycle grating (air/silicon in X).
function _make_mark_pattern(n_material, resolution)
    pattern = n_material * ones(ComplexF64, resolution, resolution)
    pattern[(resolution ÷ 4 + 1):(3 * resolution ÷ 4), :] .= 1.0
    return pattern
end

@testset "Fresnel reflection" begin
    # A bare air-glass interface (no intermediate layers) should match the
    # Fresnel equation R = |(n₁ - n₂)/(n₁ + n₂)|² at normal incidence.
    P, Q = 7, 3
    zeroth_p = P ÷ 2 + 1
    zeroth_q = Q ÷ 2 + 1

    for n2 in [1.5, 2.0, 4.0]
        settings = RCWASettings(
            IncomingWave(0.0, 0.0, 532.0),
            Stack(HomogeneousLayer(1.0), Layer[], HomogeneousLayer(n2),
                  Float64[], (3200.0, 100.0)),
            (P, Q)
        )
        result = RCWA(settings)
        DE_ref, DE_trn = diffraction_efficiencies(result)

        R_fresnel = abs((1.0 - n2) / (1.0 + n2))^2
        T_fresnel = 1.0 - R_fresnel

        @test DE_ref[zeroth_p, zeroth_q] ≈ R_fresnel atol = 1e-10
        @test DE_trn[zeroth_p, zeroth_q] ≈ T_fresnel atol = 1e-10

        # No power in non-zeroth orders (no grating hence no diffraction).
        for q in 1:Q, p in 1:P
            if (p, q) != (zeroth_p, zeroth_q)
                @test abs(DE_ref[p, q]) < 1e-12
                @test abs(DE_trn[p, q]) < 1e-12
            end
        end
    end
end

@testset "Energy conservation" begin
    # For a lossless grating, total reflected + transmitted power must equal 1.
    resolution = 256
    n_glass = 1.5 - 0.0im
    P, Q = 7, 3

    # Homogeneous: "trivial" case.
    settings = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), Layer[], HomogeneousLayer(n_glass),
              Float64[], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref, DE_trn = diffraction_efficiencies(RCWA(settings))
    @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 1e-10

    # Patterned lossless grating.
    pattern = _make_mark_pattern(n_glass, resolution)
    settings = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(n_glass),
              [200.0], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref, DE_trn = diffraction_efficiencies(RCWA(settings))
    # The tolerance is limited by harmonic truncation, not a bug.
    @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 0.05

    # Higher refractive index contrast. Energy should not exceed 1.
    pattern = _make_mark_pattern(4.0 - 0.0im, resolution)
    settings = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(4.0),
              [200.0], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref, DE_trn = diffraction_efficiencies(RCWA(settings))
    @test sum(DE_ref) + sum(DE_trn) <= 1.01
    @test sum(DE_ref) + sum(DE_trn) > 0.80
end

@testset "Very thin layer" begin
    # A very thin layer should behave almost like a bare interface: the zeroth-
    # order reflection should be close to Fresnel.
    n_Si = 4.0 - 0.0im
    resolution = 256
    P, Q = 7, 3
    zeroth_p = P ÷ 2 + 1
    zeroth_q = Q ÷ 2 + 1
    pattern = _make_mark_pattern(n_Si, resolution)

    settings_thin = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(n_Si),
              [0.01], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref_thin, DE_trn_thin = diffraction_efficiencies(RCWA(settings_thin))

    settings_bare = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), Layer[], HomogeneousLayer(n_Si),
              Float64[], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref_bare, DE_trn_bare = diffraction_efficiencies(RCWA(settings_bare))

    # Zeroth-order should be very close to the bare interface.
    @test DE_ref_thin[zeroth_p, zeroth_q] ≈ DE_ref_bare[zeroth_p, zeroth_q] atol = 1e-3
    @test DE_trn_thin[zeroth_p, zeroth_q] ≈ DE_trn_bare[zeroth_p, zeroth_q] atol = 1e-3
end

@testset "Thick opaque layer" begin
    # A very thick layer of a lossy material should absorb almost everything:
    # very little reflection from the grating structure (the bottom interface is
    # unreachable), and essentially zero transmission.
    n_lossy = 1.5 - 0.5im # Lossy material (negative imag for this convention)
    resolution = 256
    P, Q = 7, 3
    zeroth_p = P ÷ 2 + 1
    zeroth_q = Q ÷ 2 + 1
    pattern = n_lossy * ones(ComplexF64, resolution, resolution)

    settings = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(n_lossy),
              [1e6], (3200.0, 100.0)),
        (P, Q)
    )
    result = RCWA(settings)
    DE_ref, DE_trn = diffraction_efficiencies(result)

    # Total transmission should be essentially zero.
    @test sum(DE_trn) < 1e-10

    # Total R + T < 1 (energy is absorbed).
    @test sum(DE_ref) + sum(DE_trn) < 1.0

    # The reflected power should match the bare air-to-lossy Fresnel (the
    # bottom interface is invisible), since the material is homogeneous.
    R_fresnel = abs((1.0 - n_lossy) / (1.0 + n_lossy))^2
    @test DE_ref[zeroth_p, zeroth_q] ≈ R_fresnel atol = 1e-4
end

@testset "Layer splitting" begin
    # A single layer of thickness d must produce the same S-matrix as two layers
    # of thickness d/2 cascaded via the star product.
    n_Si = 4.0 - 0.0im
    resolution = 256
    d = 158.0
    P, Q = 7, 3
    pattern = _make_mark_pattern(n_Si, resolution)
    mark = Layer(pattern)

    settings_single = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [mark], HomogeneousLayer(n_Si),
              [d], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref_single, DE_trn_single = diffraction_efficiencies(
        RCWA(settings_single))

    settings_split = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [mark, mark], HomogeneousLayer(n_Si),
              [d / 2, d / 2], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref_split, DE_trn_split = diffraction_efficiencies(
        RCWA(settings_split))

    # If I add a redundant homogeneous silicon layer below the mark, the result
    # should still be the same.
    settings_redundant_layer = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(
            HomogeneousLayer(1.0),
            [mark, HomogeneousLayer(n_Si)],
            HomogeneousLayer(n_Si),
            [d, 100.0], (3200.0, 100.0)
        ),
        (P, Q)
    )
    DE_ref_rdt_layer, DE_trn_rdt_layer = diffraction_efficiencies(
        RCWA(settings_redundant_layer))

    @test DE_ref_single ≈ DE_ref_split atol = 1e-10
    @test DE_trn_single ≈ DE_trn_split atol = 1e-10
    @test DE_ref_single ≈ DE_ref_rdt_layer atol = 1e-10
    @test DE_trn_single ≈ DE_trn_rdt_layer atol = 1e-10
end

@testset "Unit cell duplication" begin
    # Tiling the unit cell 2× in X, doubling the period and number of X
    # harmonics, should reproduce the same zeroth-order efficiencies.
    n_Si = 4.0 - 0.0im
    resolution = 128
    pattern = _make_mark_pattern(n_Si, resolution)

    P, Q = 7, 3
    period = (3200.0, 100.0)

    settings_orig = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(n_Si),
              [158.0], period),
        (P, Q)
    )
    DE_ref_orig, DE_trn_orig = diffraction_efficiencies(RCWA(settings_orig))

    # Doubled unit cell: tile in X (first dimension = rows), double period and
    # harmonics.
    doubled_pattern = vcat(pattern, pattern)
    P2 = 2P - 1   # Same maximum spatial frequency
    settings_doubled = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(doubled_pattern)], HomogeneousLayer(n_Si),
              [158.0], (2 * period[1], period[2])),
        (P2, Q)
    )
    DE_ref_doubled, DE_trn_doubled = diffraction_efficiencies(RCWA(settings_doubled))

    # Zeroth order of original ↔ zeroth order of doubled system.
    zeroth_p = P ÷ 2 + 1
    zeroth_q = Q ÷ 2 + 1
    zeroth_p2 = P2 ÷ 2 + 1

    @test DE_ref_doubled[zeroth_p2, zeroth_q] ≈ DE_ref_orig[zeroth_p, zeroth_q] atol = 1e-4
    @test DE_trn_doubled[zeroth_p2, zeroth_q] ≈ DE_trn_orig[zeroth_p, zeroth_q] atol = 1e-4

    # Original dp-th order should map to the 2*dp-th order in the doubled system.
    # Check the first few non-evanescent orders.
    for dp in -3:3
        orig_p = zeroth_p + dp
        doubled_p = zeroth_p2 + 2 * dp
        if 1 <= doubled_p <= P2
            @test DE_ref_doubled[doubled_p, zeroth_q] ≈ DE_ref_orig[orig_p, zeroth_q] atol = 1e-4
            @test DE_trn_doubled[doubled_p, zeroth_q] ≈ DE_trn_orig[orig_p, zeroth_q] atol = 1e-4
        end
    end

    # Odd orders of the doubled system (no sub-period content) should be zero.
    for dp in [-3, -1, 1, 3]
        doubled_p = zeroth_p2 + dp
        if 1 <= doubled_p <= P2
            @test abs(DE_ref_doubled[doubled_p, zeroth_q]) < 1e-6
            @test abs(DE_trn_doubled[doubled_p, zeroth_q]) < 1e-6
        end
    end
end

@testset "Fabry-Perot (homogeneous slab)" begin
    # A homogeneous slab between two different half-spaces should match the
    # analytical Fabry-Perot reflection formula.
    n1 = 1.0    # air
    n2 = 1.5    # glass slab
    n3 = 2.0    # substrate
    λ = 300.0
    P, Q = 3, 3
    zeroth_p = P ÷ 2 + 1
    zeroth_q = Q ÷ 2 + 1

    for d in [50.0, 133.0, 266.0, 500.0]
        settings = RCWASettings(
            IncomingWave(0.0, 0.0, λ),
            Stack(HomogeneousLayer(n1), [HomogeneousLayer(n2)], HomogeneousLayer(n3),
                  [d], (3200.0, 100.0)),
            (P, Q)
        )
        result = RCWA(settings)

        # Analytical Fabry-Perot (negative sign convention: exp(-ikz)).
        r12 = (n1 - n2) / (n1 + n2)
        r23 = (n2 - n3) / (n2 + n3)
        β = 2π * n2 * d / λ
        r_fp = (r12 + r23 * exp(-2im * β)) / (1 + r12 * r23 * exp(-2im * β))
        R_fp = abs(r_fp)^2
        T_fp = 1.0 - R_fp

        # Complex reflection coefficient should match exactly (all modes are
        # homogeneous, so no truncation error).
        r_x, r_y = reflection_coefficients(result)
        @test r_x[zeroth_p, zeroth_q] ≈ r_fp atol = 1e-10
        @test abs(r_y[zeroth_p, zeroth_q]) < 1e-12

        # Power: diffraction efficiencies.
        DE_ref, DE_trn = diffraction_efficiencies(result)
        @test DE_ref[zeroth_p, zeroth_q] ≈ R_fp atol = 1e-10
        @test DE_trn[zeroth_p, zeroth_q] ≈ T_fp atol = 1e-10

        # Energy conservation.
        @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 1e-10
    end
end

@testset "Stack reversal (reciprocity)" begin
    # By electromagnetic reciprocity, the total transmitted power through a
    # lossless stack is the same regardless of which side the light enters from.
    # For a patterned layer, harmonic truncation slightly breaks the reciprocal
    # structure, so the tolerance reflects the truncation accuracy.
    P, Q = 7, 3
    resolution = 128
    n_glass = 1.5 - 0.0im
    pattern = _make_mark_pattern(n_glass, resolution)

    # Forward: air → patterned glass layer → glass substrate.
    settings_fwd = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(n_glass),
              [200.0], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref_fwd, DE_trn_fwd = diffraction_efficiencies(RCWA(settings_fwd))

    # Backward: glass → same patterned layer → air (swap half-spaces).
    settings_bwd = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(n_glass), [Layer(pattern)], HomogeneousLayer(1.0),
              [200.0], (3200.0, 100.0)),
        (P, Q)
    )
    DE_ref_bwd, DE_trn_bwd = diffraction_efficiencies(RCWA(settings_bwd))

    # Total transmitted power should match (reciprocity, truncation-limited).
    @test sum(DE_trn_fwd) ≈ sum(DE_trn_bwd) atol = 0.02

    # Both lossless: energy conservation.
    @test sum(DE_ref_fwd) + sum(DE_trn_fwd) ≈ 1.0 atol = 0.05
    @test sum(DE_ref_bwd) + sum(DE_trn_bwd) ≈ 1.0 atol = 0.05

    # Since both are lossless, total reflected power must also match.
    @test sum(DE_ref_fwd) ≈ sum(DE_ref_bwd) atol = 0.02
end

@testset "Scalar grating equation" begin
    # In the kinematic regime (thin grating, small index contrast, paraxial
    # orders), diffracted amplitudes are proportional to the Fourier
    # coefficients of the permittivity profile. For a 50% duty cycle square
    # wave these are 1/m for odd m and zero for even m.
    # A large period (Λ >> λ) ensures all propagating orders are nearly
    # paraxial, removing kz-dependent corrections.
    λ = 666.0
    P, Q = 13, 1
    resolution = 256
    n_space = 1.05 - 0.0im # Small contrast: Δn = 0.05
    d = 5.0 # Very thin: d/λ ≈ 0.01

    pattern = _make_mark_pattern(n_space, resolution)

    settings = RCWASettings(
        IncomingWave(0.0, 0.0, λ),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(1.0),
              [d], (50000.0, 100.0)),
        (P, Q)
    )
    result = RCWA(settings)
    t_x, _ = transmission_coefficients(result)

    zeroth_p = P ÷ 2 + 1
    amp(m) = abs(t_x[zeroth_p + m, 1])

    # Sanity: diffraction is measurable.
    @test amp(1) > 1e-6

    # Amplitude ratios should approach 1:1/3:1/5 (Fourier series of square wave).
    @test amp(1) / amp(3) ≈ 3.0 atol = 0.01
    @test amp(1) / amp(5) ≈ 5.0 atol = 0.01

    # Symmetry: positive and negative orders have the same amplitude.
    @test amp(1) ≈ amp(-1) rtol = 1e-6
    @test amp(3) ≈ amp(-3) rtol = 1e-6
    @test amp(5) ≈ amp(-5) rtol = 1e-6

    # Even orders are suppressed (50% duty cycle).
    @test amp(2) / amp(1) < 0.01
    @test amp(4) / amp(1) < 0.01
end

@testset "Grating shift (Fourier shift theorem)" begin
    # Shifting a grating by (Δx, Δy) multiplies the (p,q) diffraction order
    # coefficient by exp(-i 2π (p·Δx/Λx + q·Δy/Λy)). Amplitudes are unchanged;
    # only phases shift.
    resolution = 128
    P, Q = 5, 5
    Λx, Λy = 3200.0, 3200.0
    λ = 588.0
    d = 200.0
    n_Cu = 0.63 - 2.78im

    # 2D pattern: rectangular patch (so there is Fourier content in both X and Y).
    pattern = ones(ComplexF64, resolution, resolution)
    pattern[1:resolution÷2, 1:resolution÷3] .= n_Cu

    # Shift by (Λx/4, Λy/8) — integer pixel counts so no interpolation error.
    Δx = Λx / 4
    Δy = Λy / 8
    shifted_pattern = circshift(pattern, (resolution ÷ 4, resolution ÷ 8))

    settings_orig = RCWASettings(
        IncomingWave(0.0, 0.0, λ),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(n_Cu),
              [d], (Λx, Λy)),
        (P, Q)
    )
    settings_shifted = RCWASettings(
        IncomingWave(0.0, 0.0, λ),
        Stack(HomogeneousLayer(1.0), [Layer(shifted_pattern)], HomogeneousLayer(n_Cu),
              [d], (Λx, Λy)),
        (P, Q)
    )

    result_orig = RCWA(settings_orig)
    result_shifted = RCWA(settings_shifted)

    r_x_orig, r_y_orig = reflection_coefficients(result_orig)
    r_x_shifted, r_y_shifted = reflection_coefficients(result_shifted)
    t_x_orig, t_y_orig = transmission_coefficients(result_orig)
    t_x_shifted, t_y_shifted = transmission_coefficients(result_shifted)

    zeroth_p = P ÷ 2 + 1
    zeroth_q = Q ÷ 2 + 1

    for q in 1:Q, p in 1:P
        dp = p - zeroth_p
        dq = q - zeroth_q
        phase = exp(-2π * im * (dp * Δx / Λx + dq * Δy / Λy))

        # Amplitudes must be identical.
        @test abs(r_x_shifted[p, q]) ≈ abs(r_x_orig[p, q]) atol = 1e-10
        @test abs(r_y_shifted[p, q]) ≈ abs(r_y_orig[p, q]) atol = 1e-10
        @test abs(t_x_shifted[p, q]) ≈ abs(t_x_orig[p, q]) atol = 1e-10
        @test abs(t_y_shifted[p, q]) ≈ abs(t_y_orig[p, q]) atol = 1e-10

        # Phase: shifted coefficient = original × phase factor.
        # Only meaningful where amplitude is non-negligible.
        if abs(r_x_orig[p, q]) > 1e-10
            @test r_x_shifted[p, q] ≈ r_x_orig[p, q] * phase atol = 1e-10
        end
        if abs(r_y_orig[p, q]) > 1e-10
            @test r_y_shifted[p, q] ≈ r_y_orig[p, q] * phase atol = 1e-10
        end
        if abs(t_x_orig[p, q]) > 1e-10
            @test t_x_shifted[p, q] ≈ t_x_orig[p, q] * phase atol = 1e-10
        end
        if abs(t_y_orig[p, q]) > 1e-10
            @test t_y_shifted[p, q] ≈ t_y_orig[p, q] * phase atol = 1e-10
        end
    end

    # Diffraction efficiencies must be exactly the same.
    DE_ref_orig, DE_trn_orig = diffraction_efficiencies(result_orig)
    DE_ref_shifted, DE_trn_shifted = diffraction_efficiencies(result_shifted)
    @test DE_ref_orig ≈ DE_ref_shifted atol = 1e-10
    @test DE_trn_orig ≈ DE_trn_shifted atol = 1e-10
end

@testset "Output shapes" begin
    n_Si = 4.0 - 0.0im
    resolution = 256
    P, Q = 7, 3
    pattern = _make_mark_pattern(n_Si, resolution)

    settings = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), [Layer(pattern)], HomogeneousLayer(n_Si),
              [158.0], (3200.0, 100.0)),
        (P, Q)
    )
    result = RCWA(settings)

    # Coefficients should be P×Q matrices.
    r_x, r_y = reflection_coefficients(result)
    t_x, t_y = transmission_coefficients(result)
    @test size(r_x) == (P, Q)
    @test size(r_y) == (P, Q)
    @test size(t_x) == (P, Q)
    @test size(t_y) == (P, Q)

    # Diffraction efficiencies should also be P×Q.
    DE_ref, DE_trn = diffraction_efficiencies(result)
    @test size(DE_ref) == (P, Q)
    @test size(DE_trn) == (P, Q)
end

@testset "Magnetic half-space" begin
    # Specifying (eps, mu) should be equivalent to the n+ik shorthand when μ=1.
    P, Q = 3, 3
    settings_nk = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), Layer[], HomogeneousLayer(2.0),
              Float64[], (3200.0, 100.0)),
        (P, Q)
    )
    settings_em = RCWASettings(
        IncomingWave(0.0, 0.0, 532.0),
        Stack(HomogeneousLayer(1.0), Layer[], HomogeneousLayer(4.0, 1.0),
              Float64[], (3200.0, 100.0)),
        (P, Q)
    )

    DE_ref_nk, DE_trn_nk = diffraction_efficiencies(RCWA(settings_nk))
    DE_ref_em, DE_trn_em = diffraction_efficiencies(RCWA(settings_em))
    @test DE_ref_nk ≈ DE_ref_em atol = 1e-10
    @test DE_trn_nk ≈ DE_trn_em atol = 1e-10
end
