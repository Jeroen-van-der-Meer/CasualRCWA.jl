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

    CM = convolve(M, (1, 1))
    CM_ref = [
        0.4621 + 0.4906im
    ]
    @test all(isapprox.(CM, CM_ref; atol = 1e-4))

    CM = convolve(M, (2, 2))
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

    CM = convolve(M, (2, 3))
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

    prepared_wave_vectors = prepare_wave_vectors(
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
    layer_modes = compute_modes(E, M, Kx, Ky)

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
    A = ScatteringMatrix(A11, A12, A21, A22)
    B = ScatteringMatrix(A22, A21, A12, A11)

    S = star_product(A, B)
    T = star_product(B, A)

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

@testset "Mark in silicon" begin
    # see sf_2d_silicon.m example for reference testing

    resolution = 256

    #n_Si = 4.0 - 0.03im # Refractive index of silicon
    n_Si = 4.149970008756567 - 4.395052539404554e-02*im # at 532 nm
    n_Air = 1.0 # Refractive index of air
    top_medium = HomogeneousLayer(n_Air)

    # 50% duty cycle mark
    mark_n = n_Si * ones(ComplexF64, resolution, resolution)
    mark_n[resolution÷4+1:3*resolution÷4,:] .= n_Air
    mark_layer = Layer(mark_n) # This pattern will repeat in X and Y
    #mark_layer = HomogeneousLayer(n_Si)
    mark_thickness = 158
    bottom_medium = HomogeneousLayer(n_Si)
    period = (3200, 100) # 8 um pitch

    wavelength = 532 # Yellow light
    azimuthal_angle = 0.0 # Light comes straight down
    elevation_angle = 0.0
    incoming_wave = IncomingWave(azimuthal_angle, elevation_angle, wavelength)

    number_of_harmonics = (7, 3)

    wave_vectors = prepare_wave_vectors(
        incoming_wave,
        top_medium,
        bottom_medium,
        period,
        number_of_harmonics
    )

    top_medium = convolve(top_medium, number_of_harmonics)
    mark_layer = convolve(mark_layer, number_of_harmonics)
    bottom_medium = convolve(bottom_medium, number_of_harmonics)
    empty_medium = convolve(RCWAForRetards.EmptyLayer(), number_of_harmonics)

    top_modes = compute_modes(top_medium, wave_vectors)
    layer_modes = compute_modes(mark_layer, wave_vectors)
    bottom_modes = compute_modes(bottom_medium, wave_vectors)
    empty_modes = compute_modes(empty_medium, wave_vectors)

    # I think the only remaining delta between our code and Matlab (example1_alt)
    # is that the Matlab code uses a different formula for computing the eigenmodes
    # of free space.
    S1 = compute_global_scattering_matrix(
        top_modes, LayerModes[], bottom_modes, empty_modes,
        wavelength, Float64[]
    )
    S2 = compute_global_scattering_matrix(
        top_modes, LayerModes[layer_modes], bottom_modes, empty_modes,
        wavelength, Float64[mark_thickness]
    )
    S3 = compute_global_scattering_matrix(
        top_modes, [layer_modes, layer_modes], bottom_modes, empty_modes,
        wavelength, [mark_thickness / 2, mark_thickness / 2]
    )
    # Not done yet
end