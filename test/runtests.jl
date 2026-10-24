#!/usr/bin/env julia

using CasualRCWA
using LinearAlgebra
using Test

@testset "Convolution 4x4" begin
    M = [
        0.8825 + 0.8059im   0.1837 + 0.1019im   0.5598 + 0.1879im   0.6498 + 0.4685im
        0.5390 + 0.5389im   0.2492 + 0.6446im   0.4151 + 0.0094im   0.2888 + 0.9085im
        0.5437 + 0.5901im   0.2851 + 0.9801im   0.9057 + 0.8383im   0.2553 + 0.4171im
        0.4425 + 0.2311im   0.5295 + 0.1017im   0.3051 + 0.4813im   0.3582 + 0.5441im
    ]

    CM = CasualRCWA.convolve(M, (1, 1))
    CM_ref = [
        0.4621 + 0.4906im
    ]
    @test all(isapprox.(CM, CM_ref; atol = 1e-4))

    CM = CasualRCWA.convolve(M, (2, 2))
    CM_ref = [
        0.4621 + 0.4906im   0.0377 - 0.0741im   0.0154 + 0.0428im  -0.0090 + 0.0530im
       -0.0035 - 0.0854im   0.4621 + 0.4906im   0.1106 + 0.0285im   0.0154 + 0.0428im
        0.0265 + 0.0494im  -0.0626 + 0.0774im   0.4621 + 0.4906im   0.0377 - 0.0741im
        0.0606 - 0.0378im   0.0265 + 0.0494im  -0.0035 - 0.0854im   0.4621 + 0.4906im
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

    CM = CasualRCWA.convolve(M, (2, 3))
    CM_ref = [
        0.4734 + 0.5013im   0.0013 - 0.0450im   0.0336 + 0.1460im  -0.0682 - 0.0198im  -0.0278 + 0.0172im  -0.0908 - 0.0124im
       -0.0281 + 0.0264im   0.4734 + 0.5013im  -0.0138 - 0.0419im   0.0336 + 0.1460im   0.0368 + 0.0158im  -0.0278 + 0.0172im
        0.0809 + 0.0082im  -0.1113 + 0.0349im   0.4734 + 0.5013im   0.0013 - 0.0450im   0.0336 + 0.1460im  -0.0682 - 0.0198im
       -0.0415 + 0.0746im   0.0809 + 0.0082im  -0.0281 + 0.0264im   0.4734 + 0.5013im  -0.0138 - 0.0419im   0.0336 + 0.1460im
       -0.0163 - 0.0081im  -0.0564 - 0.0380im   0.0809 + 0.0082im  -0.1113 + 0.0349im   0.4734 + 0.5013im   0.0013 - 0.0450im
        0.0225 + 0.0008im  -0.0163 - 0.0081im  -0.0415 + 0.0746im   0.0809 + 0.0082im  -0.0281 + 0.0264im   0.4734 + 0.5013im
    ]
    @test all(isapprox.(CM, CM_ref; atol = 1e-4))
end

@testset "Wave vector preparation" begin
    wavelength = 0.02
    elevation_angle = 1.0
    azimuthal_angle = 0.5
    source = Source(azimuthal_angle, elevation_angle, wavelength)

    top_medium = Layer(2.0, 1.0)
    bottom_medium = Layer(9.0, 1.0)
    period = (0.0175, 0.015)

    number_of_harmonics = (3, 3)

    prepared_wave_vectors = CasualRCWA.prepare_wave_vectors(
        source,
        top_medium,
        bottom_medium,
        period,
        number_of_harmonics
    )

    # Orders run from -1 to +1 with X varying fastest; order m has
    # K_x = k_x + m λ/Λx.
    K_x = diag(prepared_wave_vectors.waveVectorsX)
    K_x_ref = [
       -0.0985 + 0.0000im
        1.0443 + 0.0000im
        2.1872 + 0.0000im
       -0.0985 + 0.0000im
        1.0443 + 0.0000im
        2.1872 + 0.0000im
       -0.0985 + 0.0000im
        1.0443 + 0.0000im
        2.1872 + 0.0000im
    ]
    @test all(isapprox.(K_x, K_x_ref; atol = 1e-4))

    K_y = diag(prepared_wave_vectors.waveVectorsY)
    K_y_ref = [
       -0.7628 + 0.0000im
       -0.7628 + 0.0000im
       -0.7628 + 0.0000im
        0.5705 + 0.0000im
        0.5705 + 0.0000im
        0.5705 + 0.0000im
        1.9039 + 0.0000im
        1.9039 + 0.0000im
        1.9039 + 0.0000im
    ]
    @test all(isapprox.(K_y, K_y_ref; atol = 1e-4))

    # Reflected waves travel (or decay) in the -z direction.
    K_reflection = diag(prepared_wave_vectors.waveVectorsZReflection)
    K_reflection_ref = [
       -1.1868 + 0.0000im
       -0.5723 + 0.0000im
        0.0000 - 1.8346im
       -1.2903 + 0.0000im
       -0.7641 + 0.0000im
        0.0000 - 1.7633im
        0.0000 - 1.2784im
        0.0000 - 1.6478im
        0.0000 - 2.5315im
    ]
    @test all(isapprox.(K_reflection, K_reflection_ref; atol = 1e-4))

    K_transmission = diag(prepared_wave_vectors.waveVectorsZTransmission)
    K_transmission_ref = [
        2.8997 + 0.0000im
        2.7069 + 0.0000im
        1.9064 + 0.0000im
        2.9436 + 0.0000im
        2.7539 + 0.0000im
        1.9725 + 0.0000im
        2.3164 + 0.0000im
        2.0699 + 0.0000im
        0.7691 + 0.0000im
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
    layer_modes = CasualRCWA.compute_modes(E, M, Kx, Ky)

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
    R, T = CasualRCWA._compute_symmetric_scattering_matrix(A, B, X)

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
    A = CasualRCWA.ScatteringMatrix(A11, A12, A21, A22)
    B = CasualRCWA.ScatteringMatrix(A22, A21, A12, A11)

    S = CasualRCWA.star_product(A, B)
    T = CasualRCWA.star_product(B, A)

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
    incoming = Source(0.0, 0.0, 532.0)
    top = Layer(1.0)
    bottom = Layer(1.0)
    nh = (3, 3)
    wv = CasualRCWA.prepare_wave_vectors(incoming, top, bottom, (3200.0, 100.0), nh)

    empty_c = CasualRCWA.convolve(Layer(1.0), nh)
    modes = CasualRCWA.compute_modes(empty_c, wv)

    PQ = prod(nh)
    @test modes.electricModes ≈ Matrix{ComplexF64}(I, 2PQ, 2PQ)

    # Eigenvalues should satisfy the dispersion relation λ = -i kz with
    # kz = sqrt(εμ - kx² - ky²), where the forward wave exp(ikz z) propagates
    # (real kz > 0) or decays (imaginary kz with positive imaginary part).
    kx = real(diag(wv.waveVectorsX))
    ky = real(diag(wv.waveVectorsY))
    kz2 = 1.0 .- kx.^2 .- ky.^2
    kz_expected = [k >= 0 ? complex(sqrt(k)) : im * sqrt(-k) for k in kz2]
    λ_expected = -im .* kz_expected
    @test modes.eigenvalues ≈ vcat(λ_expected, λ_expected)
end

@testset "Homogeneous modes: branch independent of signed zeros" begin
    # Regression test: in 2D, orders with kx < 0 and ky < 0 have kx² + ky²
    # with imaginary part -0.0 (e.g. (-0.5 + 0.0im)^2 == 0.25 - 0.0im). This
    # used to put the square root on the wrong side of its branch cut, swapping
    # forward and backward waves for those orders and breaking energy
    # conservation for all genuinely 2D patterns.
    @test (-0.5 + 0.0im)^2 === 0.25 - 0.0im # The culprit.
    for s in (0.0, -0.0)
        # Propagating: positive real kz regardless of the sign of zero.
        @test CasualRCWA._forward_kz(complex(0.39, s)) ≈ sqrt(0.39)
        @test real(CasualRCWA._forward_kz(complex(0.39, s))) > 0
        # Evanescent: decaying in +z, so positive imaginary kz.
        @test CasualRCWA._forward_kz(complex(-0.39, s)) ≈ im * sqrt(0.39)
    end
    # Absorbing: decaying in +z.
    @test imag(CasualRCWA._forward_kz(2.25 + 0.1im)) > 0
    @test real(CasualRCWA._forward_kz(2.25 + 0.1im)) > 0

    wv = CasualRCWA.prepare_wave_vectors(
        Source(0.6), Layer(1.0), Layer(1.5), (1.2, 1.0), (3, 3)
    )
    modes = CasualRCWA.compute_modes(CasualRCWA.convolve(Layer(1.0), (3, 3)), wv)
    kz_air = -diag(wv.waveVectorsZReflection)
    # Every harmonic in this setup propagates in air.
    @test all(abs.(imag.(kz_air)) .< 1e-15) && all(real.(kz_air) .> 0)
    @test modes.eigenvalues ≈ vcat(-im .* kz_air, -im .* kz_air)
    @test all(real.(diag(wv.waveVectorsZTransmission)) .> 0)
end

# Common pattern for tests: 50% duty cycle grating (air/silicon in X).
# Convention: columns = X, rows = Y (top = +Y).
function _make_mark_pattern(n_material, resolution)
    pattern = n_material * ones(ComplexF64, resolution, resolution)
    pattern[:, (resolution ÷ 4 + 1):(3 * resolution ÷ 4)] .= 1.0
    return pattern
end

@testset "Fresnel reflection" begin
    N, M = 3, 1
    P = 2N + 1
    Q = 2M + 1
    zeroth_p = N + 1
    zeroth_q = M + 1

    @testset "Normal incidence" begin
        # R = |(n₁ - n₂)/(n₁ + n₂)|²
        for n2 in [1.5, 2.0, 4.0]
            input = RCWAInput(
                Source(0.0, 0.0, 532.0),
                Stack([Layer(1.0), Layer(n2)],
                      [Inf, Inf], (3200.0, 100.0)),
                (N, M)
            )
            result = RCWA(input)
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

    @testset "Oblique incidence (ϕ = 0)" begin
        # With ϕ=0 the plane of incidence is x-z. Then:
        # - polarization :y = s-pol (TE)
        # - polarization :x → p-pol (TM)
        # Fresnel formulas:
        # - r_s = (n₁ cos θ_i - n₂ cos θ_t) / (n₁ cos θ_i + n₂ cos θ_t)
        # - r_p = (n₂ cos θ_i - n₁ cos θ_t) / (n₂ cos θ_i + n₁ cos θ_t)
        n1 = 1.0
        for (n2, θ_i) in [(1.5, 0.3), (2.0, 0.5), (4.0, 0.15), (1.5, 1.0)]
            cos_i = cos(θ_i)
            cos_t = sqrt(1 - (n1 / n2 * sin(θ_i))^2)   # complex if TIR

            r_s = (n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t)
            r_p = (n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t)

            input = RCWAInput(
                Source(0.0, θ_i, 532.0),
                Stack([Layer(n1), Layer(n2)],
                      [Inf, Inf], (3200.0, 100.0)),
                (N, M)
            )

            # s-polarization (TE, :y)
            DE_ref_s, DE_trn_s = diffraction_efficiencies(RCWA(input);
                                                          polarization = :y)
            R_s = abs(r_s)^2
            @test DE_ref_s[zeroth_p, zeroth_q] ≈ R_s atol = 1e-10
            @test sum(DE_ref_s) + sum(DE_trn_s) ≈ 1.0 atol = 1e-10

            # p-polarization (TM, :x)
            DE_ref_p, DE_trn_p = diffraction_efficiencies(RCWA(input);
                                                          polarization = :x)
            R_p = abs(r_p)^2
            @test DE_ref_p[zeroth_p, zeroth_q] ≈ R_p atol = 1e-10
            @test sum(DE_ref_p) + sum(DE_trn_p) ≈ 1.0 atol = 1e-10

            # No power in non-zeroth orders.
            for q in 1:Q, p in 1:P
                if (p, q) != (zeroth_p, zeroth_q)
                    @test abs(DE_ref_s[p, q]) < 1e-12
                    @test abs(DE_ref_p[p, q]) < 1e-12
                end
            end
        end
    end

    @testset "s/p symbols and Jones vector" begin
        n1, n2 = 1.0, 1.5

        # ϕ = 0: :s should match :y, :p should match :x
        for θ_i in [0.0, 0.3, 0.8]
            input = RCWAInput(
                Source(0.0, θ_i, 532.0),
                Stack([Layer(n1), Layer(n2)], [Inf, Inf], (3200.0, 100.0)),
                (N, M)
            )
            result = RCWA(input)

            DE_x, _ = diffraction_efficiencies(result; polarization = :x)
            DE_p, _ = diffraction_efficiencies(result; polarization = :p)
            @test DE_x ≈ DE_p

            _, DE_y = diffraction_efficiencies(result; polarization = :y)
            _, DE_s = diffraction_efficiencies(result; polarization = :s)
            @test DE_y ≈ DE_s

            # Jones vector [1, 0] should match :x
            DE_jones, _ = diffraction_efficiencies(result; polarization = [1, 0])
            @test DE_x ≈ DE_jones

            # Unnormalized vector should still match
            DE_1234, _ = diffraction_efficiencies(result; polarization = [1234, 0])
            @test DE_x ≈ DE_1234
        end

        # Oblique incidence with nonzero ϕ: :s and :p should still satisfy
        # Fresnel power conservation and give correct reflectance.
        for (ϕ, θ_i) in [(π / 4, 0.3), (π / 3, 0.5), (1.2, 0.15)]
            cos_i = cos(θ_i)
            cos_t = sqrt(1 - (n1 / n2 * sin(θ_i))^2)

            R_s = abs((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t))^2
            R_p = abs((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t))^2

            input = RCWAInput(
                Source(ϕ, θ_i, 532.0),
                Stack([Layer(n1), Layer(n2)], [Inf, Inf], (3200.0, 100.0)),
                (N, M)
            )
            result = RCWA(input)

            DE_ref_s, DE_trn_s = diffraction_efficiencies(result; polarization = :s)
            @test DE_ref_s[zeroth_p, zeroth_q] ≈ R_s atol = 1e-10
            @test sum(DE_ref_s) + sum(DE_trn_s) ≈ 1.0 atol = 1e-10

            DE_ref_p, DE_trn_p = diffraction_efficiencies(result; polarization = :p)
            @test DE_ref_p[zeroth_p, zeroth_q] ≈ R_p atol = 1e-10
            @test sum(DE_ref_p) + sum(DE_trn_p) ≈ 1.0 atol = 1e-10
        end
    end
end

@testset "Snell's law" begin
    # The wave vectors produced by RCWA encode the diffraction angles. For a
    # bare interface (no grating layers), the zeroth-order transmitted kz must
    # satisfy Snell's law: n₁ sin θ_i = n₂ sin θ_t. More generally, the
    # dispersion relation kx² + ky² + kz² = n² must hold for every order in
    # both half-spaces.
    N, M = 3, 1
    P = 2N + 1
    Q = 2M + 1
    λ = 532.0

    for (n1, n2, ϕ, θ_i) in [
        (1.0, 1.5, 0.0, 0.3),
        (1.0, 2.0, 0.0, 0.7),
        (1.5, 1.0, 0.0, 0.4),
        (1.0, 1.5, 0.6, 0.3) # Bonus test: non-zero azimuthal angle
    ]
        input = RCWAInput(
            Source(ϕ, θ_i, λ),
            Stack([Layer(n1), Layer(n2)],
                  [Inf, Inf], (3200.0, 100.0)),
            (N, M)
        )
        result = RCWA(input)
        wv = result.waveVectors

        kx = diag(wv.waveVectorsX)
        ky = diag(wv.waveVectorsY)
        kz_top = diag(wv.waveVectorsZReflection)
        kz_bot = diag(wv.waveVectorsZTransmission)

        # Dispersion relation in both media: kx² + ky² + kz² = εμ = n².
        @test kx.^2 .+ ky.^2 .+ kz_top.^2 ≈ fill(complex(n1^2), P * Q) atol = 1e-10
        @test kx.^2 .+ ky.^2 .+ kz_bot.^2 ≈ fill(complex(n2^2), P * Q) atol = 1e-10

        # Zeroth-order transverse momenta: kx = n₁ sin θ cos ϕ, ky = n₁ sin θ sin ϕ.
        zeroth = N + 1 + M * P
        @test real(kx[zeroth]) ≈ n1 * sin(θ_i) * cos(ϕ) atol = 1e-10
        @test real(ky[zeroth]) ≈ n1 * sin(θ_i) * sin(ϕ) atol = 1e-10

        # Snell's law for the zeroth transmitted order: kz = sqrt(n² - kx² - ky²).
        kx0 = n1 * sin(θ_i) * cos(ϕ)
        ky0 = n1 * sin(θ_i) * sin(ϕ)
        @test real(kz_bot[zeroth]) ≈ sqrt(n2^2 - kx0^2 - ky0^2) atol = 1e-10
    end
end

@testset "Energy conservation" begin
    # For a lossless grating, total reflected + transmitted power must equal 1.
    resolution = 256
    n_glass = 1.5 + 0.0im
    N, M = 3, 1

    # Homogeneous: "trivial" case.
    input = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(n_glass)],
              [Inf, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref, DE_trn = diffraction_efficiencies(RCWA(input))
    @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 1e-10

    # Patterned lossless grating.
    pattern = _make_mark_pattern(n_glass, resolution)
    input = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(pattern), Layer(n_glass)],
              [Inf, 200.0, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref, DE_trn = diffraction_efficiencies(RCWA(input))
    # The tolerance is limited by harmonic truncation, not a bug.
    @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 0.05

    # Higher refractive index contrast. Energy should not exceed 1.
    pattern = _make_mark_pattern(4.0 + 0.0im, resolution)
    input = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(pattern), Layer(4.0)],
              [Inf, 200.0, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref, DE_trn = diffraction_efficiencies(RCWA(input))
    @test sum(DE_ref) + sum(DE_trn) <= 1.01
    @test sum(DE_ref) + sum(DE_trn) > 0.80
end

@testset "Very thin layer" begin
    # A very thin layer should behave almost like a bare interface: the zeroth-
    # order reflection should be close to Fresnel.
    n_Si = 4.0 + 0.0im
    resolution = 256
    N, M = 3, 1
    zeroth_p = N + 1
    zeroth_q = M + 1
    pattern = _make_mark_pattern(n_Si, resolution)

    input_thin = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(pattern), Layer(n_Si)],
              [Inf, 0.01, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref_thin, DE_trn_thin = diffraction_efficiencies(RCWA(input_thin))

    input_bare = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(n_Si)],
              [Inf, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref_bare, DE_trn_bare = diffraction_efficiencies(RCWA(input_bare))

    # Zeroth-order should be very close to the bare interface.
    @test DE_ref_thin[zeroth_p, zeroth_q] ≈ DE_ref_bare[zeroth_p, zeroth_q] atol = 1e-3
    @test DE_trn_thin[zeroth_p, zeroth_q] ≈ DE_trn_bare[zeroth_p, zeroth_q] atol = 1e-3
end

@testset "Thick opaque layer" begin
    # A very thick layer of a lossy material should absorb almost everything:
    # very little reflection from the grating structure (the bottom interface is
    # unreachable), and essentially zero transmission.
    n_lossy = 1.5 + 0.5im # Lossy material (positive imaginary part)
    resolution = 256
    N, M = 3, 1
    zeroth_p = N + 1
    zeroth_q = M + 1
    pattern = n_lossy * ones(ComplexF64, resolution, resolution)

    input = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(pattern), Layer(n_lossy)],
              [Inf, 1e6, Inf], (3200.0, 100.0)),
        (N, M)
    )
    result = RCWA(input)
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
    n_Si = 4.0 + 0.0im
    resolution = 256
    d = 158.0
    N, M = 3, 1
    pattern = _make_mark_pattern(n_Si, resolution)
    mark = Layer(pattern)

    input_single = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), mark, Layer(n_Si)],
              [Inf, d, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref_single, DE_trn_single = diffraction_efficiencies(
        RCWA(input_single))

    input_split = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), mark, mark, Layer(n_Si)],
              [Inf, d / 2, d / 2, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref_split, DE_trn_split = diffraction_efficiencies(
        RCWA(input_split))

    # If I add a redundant homogeneous silicon layer below the mark, the result
    # should still be the same.
    input_redundant_layer = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack(
            [Layer(1.0), mark, Layer(n_Si), Layer(n_Si)],
            [Inf, d, 100.0, Inf], (3200.0, 100.0)
        ),
        (N, M)
    )
    DE_ref_rdt_layer, DE_trn_rdt_layer = diffraction_efficiencies(
        RCWA(input_redundant_layer))

    @test DE_ref_single ≈ DE_ref_split atol = 1e-10
    @test DE_trn_single ≈ DE_trn_split atol = 1e-10
    @test DE_ref_single ≈ DE_ref_rdt_layer atol = 1e-10
    @test DE_trn_single ≈ DE_trn_rdt_layer atol = 1e-10
end

@testset "Unit cell duplication" begin
    # Tiling the unit cell 2× in X, doubling the period and number of X
    # harmonics, should reproduce the same zeroth-order efficiencies.
    n_Si = 4.0 + 0.0im
    resolution = 128
    pattern = _make_mark_pattern(n_Si, resolution)

    N, M = 3, 1
    period = (3200.0, 100.0)

    input_orig = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(pattern), Layer(n_Si)],
              [Inf, 158.0, Inf], period),
        (N, M)
    )
    DE_ref_orig, DE_trn_orig = diffraction_efficiencies(RCWA(input_orig))

    # Doubled unit cell: tile in X (columns), double period and harmonics.
    doubled_pattern = hcat(pattern, pattern)
    N2 = 2N   # Same maximum spatial frequency (doubled period → halved k-spacing)
    input_doubled = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(doubled_pattern), Layer(n_Si)],
              [Inf, 158.0, Inf], (2 * period[1], period[2])),
        (N2, M)
    )
    DE_ref_doubled, DE_trn_doubled = diffraction_efficiencies(RCWA(input_doubled))

    P = 2N + 1
    P2 = 2N2 + 1
    Q = 2M + 1

    # Zeroth order of original ↔ zeroth order of doubled system.
    zeroth_p = N + 1
    zeroth_q = M + 1
    zeroth_p2 = N2 + 1

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
    N, M = 1, 1
    P = 2N + 1
    Q = 2M + 1
    zeroth_p = N + 1
    zeroth_q = M + 1

    @testset "Normal incidence" begin
        for d in [50.0, 133.0, 266.0, 500.0]
            input = RCWAInput(
                Source(0.0, 0.0, λ),
                Stack([Layer(n1), Layer(n2), Layer(n3)],
                      [Inf, d, Inf], (3200.0, 100.0)),
                (N, M)
            )
            result = RCWA(input)

            # Analytical Fabry-Perot (forward waves written as exp(ikz)).
            r12 = (n1 - n2) / (n1 + n2)
            r23 = (n2 - n3) / (n2 + n3)
            β = 2π * n2 * d / λ
            r_fp = (r12 + r23 * exp(2im * β)) / (1 + r12 * r23 * exp(2im * β))
            R_fp = abs(r_fp)^2
            T_fp = 1.0 - R_fp

            # Complex reflection coefficient should match exactly (all modes are
            # homogeneous, so no truncation error).
            r_x, r_y = reflection_coefficients(result; polarization = :p)
            @test r_x[zeroth_p, zeroth_q] ≈ r_fp atol = 1e-10
            @test abs(r_y[zeroth_p, zeroth_q]) < 1e-12

            # Power: diffraction efficiencies.
            DE_ref, DE_trn = diffraction_efficiencies(result; polarization = :p)
            @test DE_ref[zeroth_p, zeroth_q] ≈ R_fp atol = 1e-10
            @test DE_trn[zeroth_p, zeroth_q] ≈ T_fp atol = 1e-10

            # Energy conservation.
            @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 1e-10
        end
    end

    @testset "Oblique incidence (sign convention test)" begin
        # Sign conventions for this code (I think):
        #   s-pol (y-input): r_y = r_s_FP  (standard Fresnel)
        #   p-pol (x-input): r_x = -r_p_FP (Ex (transverse) flips sign on reflection)
        # This test ensures we make an even amount of sign errors.
        for (θ_i, d) in [(0.3, 80.0), (0.5, 133.0), (0.8, 200.0)]
            cos_1 = cos(θ_i)
            cos_2 = sqrt(1 - (n1 * sin(θ_i) / n2)^2)
            cos_3 = sqrt(1 - (n1 * sin(θ_i) / n3)^2)
            β = 2π * n2 * cos_2 * d / λ
            X2 = exp(2im * β)

            input = RCWAInput(
                Source(0.0, θ_i, λ),
                Stack([Layer(n1), Layer(n2), Layer(n3)],
                      [Inf, d, Inf], (3200.0, 100.0)),
                (N, M)
            )
            result = RCWA(input)

            # s-polarization (TE): standard Fresnel interface coefficients.
            r12_s = (n1 * cos_1 - n2 * cos_2) / (n1 * cos_1 + n2 * cos_2)
            r23_s = (n2 * cos_2 - n3 * cos_3) / (n2 * cos_2 + n3 * cos_3)
            r_s_fp = (r12_s + r23_s * X2) / (1 + r12_s * r23_s * X2)
            _, r_y = reflection_coefficients(result; polarization = :y)
            @test r_y[zeroth_p, zeroth_q] ≈ r_s_fp atol = 1e-10

            # p-polarization (TM): the RCWA transverse-Ex reflection picks up
            # a minus sign relative to the standard Fresnel r_p at each
            # interface, so the total FP reflection is negated.
            r12_p = (n2 * cos_1 - n1 * cos_2) / (n2 * cos_1 + n1 * cos_2)
            r23_p = (n3 * cos_2 - n2 * cos_3) / (n3 * cos_2 + n2 * cos_3)
            r_p_fp = (r12_p + r23_p * X2) / (1 + r12_p * r23_p * X2)
            r_x, _ = reflection_coefficients(result; polarization = :x)
            @test r_x[zeroth_p, zeroth_q] ≈ -r_p_fp atol = 1e-10

            # Energy conservation for both polarizations.
            DE_ref_s, DE_trn_s = diffraction_efficiencies(result; polarization = :y)
            @test sum(DE_ref_s) + sum(DE_trn_s) ≈ 1.0 atol = 1e-10
            DE_ref_p, DE_trn_p = diffraction_efficiencies(result; polarization = :x)
            @test sum(DE_ref_p) + sum(DE_trn_p) ≈ 1.0 atol = 1e-10
        end
    end
end

@testset "Stack reversal (reciprocity)" begin
    # By electromagnetic reciprocity, the total transmitted power through a
    # lossless stack is the same regardless of which side the light enters from.
    # For a patterned layer, harmonic truncation slightly breaks the reciprocal
    # structure, so the tolerance reflects the truncation accuracy.
    N, M = 3, 1
    resolution = 128
    n_glass = 1.5 + 0.0im
    pattern = _make_mark_pattern(n_glass, resolution)

    # Forward: air -> patterned glass layer -> glass substrate.
    input_fwd = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(pattern), Layer(n_glass)],
              [Inf, 200.0, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref_fwd, DE_trn_fwd = diffraction_efficiencies(RCWA(input_fwd))

    # Backward: glass -> same patterned layer -> air (swap half-spaces).
    input_bwd = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(n_glass), Layer(pattern), Layer(1.0)],
              [Inf, 200.0, Inf], (3200.0, 100.0)),
        (N, M)
    )
    DE_ref_bwd, DE_trn_bwd = diffraction_efficiencies(RCWA(input_bwd))

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
    N, M = 12, 0
    P = 2N + 1
    resolution = 256
    n_space = 1.05 + 0.0im # Small contrast: Δn = 0.05
    d = 5.0 # Very thin: d/λ ≈ 0.01

    pattern = _make_mark_pattern(n_space, resolution)

    input = RCWAInput(
        Source(0.0, 0.0, λ),
        Stack([Layer(1.0), Layer(pattern), Layer(1.0)],
              [Inf, d, Inf], (50000.0, 100.0)),
        (N, M)
    )
    result = RCWA(input)
    t_x, _ = transmission_coefficients(result; polarization = :x)

    zeroth_p = N + 1
    amp(m) = abs(t_x[zeroth_p + m, 1])

    # Sanity: diffraction is measurable.
    @test amp(1) > 1e-6

    # Amplitude ratios should approach 1:1/3:1/5 (Fourier series of square wave).
    @test amp(1) / amp(3) ≈ 3.0 atol = 0.01
    @test amp(1) / amp(5) ≈ 5.0 atol = 0.02

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
    N, M = 2, 2
    P = 2N + 1
    Q = 2M + 1
    Λx, Λy = 3200.0, 3200.0
    λ = 588.0
    d = 200.0
    n_Cu = 0.63 + 2.78im

    # 2D pattern: rectangular patch (so there is Fourier content in both X and Y).
    # Convention: columns = X, rows = Y (top = +Y).
    pattern = ones(ComplexF64, resolution, resolution)
    pattern[1:resolution÷2, 1:resolution÷3] .= n_Cu

    # Shift by (Δx, Δy) = (Λx/4, Λy/8) — integer pixel counts so no
    # interpolation error.  Columns = X (shift right = +X), rows = Y with
    # top = +Y (shift up = +Y → negative row circshift).
    Δx = Λx / 4
    Δy = Λy / 8
    shifted_pattern = circshift(pattern, (-(resolution ÷ 8), resolution ÷ 4))

    input_orig = RCWAInput(
        Source(0.0, 0.0, λ),
        Stack([Layer(1.0), Layer(pattern), Layer(n_Cu)],
              [Inf, d, Inf], (Λx, Λy)),
        (N, M)
    )
    input_shifted = RCWAInput(
        Source(0.0, 0.0, λ),
        Stack([Layer(1.0), Layer(shifted_pattern), Layer(n_Cu)],
              [Inf, d, Inf], (Λx, Λy)),
        (N, M)
    )

    result_orig = RCWA(input_orig)
    result_shifted = RCWA(input_shifted)

    r_x_orig, r_y_orig = reflection_coefficients(result_orig)
    r_x_shifted, r_y_shifted = reflection_coefficients(result_shifted)
    t_x_orig, t_y_orig = transmission_coefficients(result_orig)
    t_x_shifted, t_y_shifted = transmission_coefficients(result_shifted)

    zeroth_p = N + 1
    zeroth_q = M + 1

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
    n_Si = 4.0 + 0.0im
    resolution = 256
    N, M = 3, 1
    P = 2N + 1
    Q = 2M + 1
    pattern = _make_mark_pattern(n_Si, resolution)

    input = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(pattern), Layer(n_Si)],
              [Inf, 158.0, Inf], (3200.0, 100.0)),
        (N, M)
    )
    result = RCWA(input)

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
    N, M = 1, 1
    input_nk = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(2.0)],
              [Inf, Inf], (3200.0, 100.0)),
        (N, M)
    )
    input_em = RCWAInput(
        Source(0.0, 0.0, 532.0),
        Stack([Layer(1.0), Layer(4.0, 1.0)],
              [Inf, Inf], (3200.0, 100.0)),
        (N, M)
    )

    DE_ref_nk, DE_trn_nk = diffraction_efficiencies(RCWA(input_nk))
    DE_ref_em, DE_trn_em = diffraction_efficiencies(RCWA(input_em))
    @test DE_ref_nk ≈ DE_ref_em atol = 1e-10
    @test DE_trn_nk ≈ DE_trn_em atol = 1e-10
end

@testset "Cartesian product" begin
    # If input geometry is symmetric in X and Y, then there should be a
    # corresponding symmetry in the scattering matrices.

    nk = [2+1im 1; 1 1]
    air = Layer(1.0)
    layer = Layer(nk)

    thickness = 100
    cell_size = (1000, 1000)

    wave = Source(0, 0, 532)
    stack = Stack([air, layer, air], [Inf, thickness, Inf], cell_size)
    N, M = 3, 3

    input = RCWAInput(wave, stack, (N, M))
    results = RCWA(input)

    de_ref_X, de_trn_X = diffraction_efficiencies(results; polarization = :x)
    de_ref_Y, de_trn_Y = diffraction_efficiencies(results; polarization = :y)
    @test de_ref_X ≈ transpose(de_ref_Y)
    @test de_trn_X ≈ transpose(de_trn_Y)

    # Input size of matrices shouldn't matter for outcome.

    layer_repmat = Layer(repeat(nk; inner = (10, 10)))
    stack_repmat = Stack([air, layer_repmat, air], [Inf, thickness, Inf], cell_size)

    input_repmat = RCWAInput(wave, stack_repmat, (N, M))

    results_repmat = RCWA(input_repmat)

    de_ref_X_repmat, de_trn_X_repmat =
        diffraction_efficiencies(results_repmat; polarization = :x)

    @test de_ref_X ≈ de_ref_X_repmat
    @test de_trn_X ≈ de_trn_X_repmat
end

@testset "Rayleigh anomaly" begin
    # When λ/period is an integer, a diffraction order goes exactly grazing
    # (kz = 0). This used to produce NaN/Inf in the scattering matrix. The
    # simulation should still produce finite, physical results.
    n_si = 4.0
    input = RCWAInput(
        Source(0.0, 0.0, 1.0),
        Stack([Layer(1.0), Layer([1.0 n_si n_si 1.0]), Layer(n_si)],
              [Inf, 0.5, Inf], (1.0, 1.0)),
        (10, 0)
    )
    result = RCWA(input)

    for pol in [:x, :y]
        DE_ref, DE_trn = diffraction_efficiencies(result; polarization = pol)

        # All efficiencies must be finite.
        @test all(isfinite, DE_ref)
        @test all(isfinite, DE_trn)

        # Energy conservation.
        @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 1e-6
    end
end

@testset "0D convenience constructors" begin
    # Stack without period (all homogeneous layers).
    stack_0d = Stack([Layer(1.0), Layer(1.5), Layer(2.0)], [Inf, 100.0, Inf])
    @test stack_0d.period == (1.0, 1.0)

    # RCWAInput without order.
    input_0d = RCWAInput(Source(532.0), stack_0d)
    @test input_0d.order == (0, 0)

    # Should produce the same result as the explicit version.
    input_explicit = RCWAInput(
        Source(532.0),
        Stack([Layer(1.0), Layer(1.5), Layer(2.0)], [Inf, 100.0, Inf], (1.0, 1.0)),
        (0, 0)
    )
    DE_ref_0d, DE_trn_0d = diffraction_efficiencies(RCWA(input_0d))
    DE_ref_ex, DE_trn_ex = diffraction_efficiencies(RCWA(input_explicit))
    @test DE_ref_0d ≈ DE_ref_ex
    @test DE_trn_0d ≈ DE_trn_ex
end

@testset "1D convenience constructors" begin
    # X-varying grating: columns = X in user convention.
    # A 1×4 row matrix varies in X (4 columns).
    pattern_x = [1.0+0im 1.5+0im 1.5+0im 1.0+0im]
    stack_1d_x = Stack(
        [Layer(1.0), Layer(pattern_x), Layer(1.5)],
        [Inf, 100.0, Inf], 3200.0
    )
    @test stack_1d_x.period == (3200.0, 1.0)

    input_1d_x = RCWAInput(Source(532.0), stack_1d_x, 3)
    @test input_1d_x.order == (3, 0)

    # Y-varying grating: a 4×1 matrix varies in Y (4 rows, 1 column).
    pattern_y = reshape([1.0+0im, 1.5+0im, 1.5+0im, 1.0+0im], 4, 1)
    stack_1d_y = Stack(
        [Layer(1.0), Layer(pattern_y), Layer(1.0)],
        [Inf, 100.0, Inf], 3200.0
    )
    @test stack_1d_y.period == (1.0, 3200.0)

    input_1d_y = RCWAInput(Source(532.0), stack_1d_y, 3)
    @test input_1d_y.order == (0, 3)

    # Results should match the explicit 2-tuple version.
    input_explicit_x = RCWAInput(
        Source(532.0),
        Stack([Layer(1.0), Layer(pattern_x), Layer(1.5)],
              [Inf, 100.0, Inf], (3200.0, 1.0)),
        (3, 0)
    )
    DE_ref_1d, DE_trn_1d = diffraction_efficiencies(RCWA(input_1d_x))
    DE_ref_ex, DE_trn_ex = diffraction_efficiencies(RCWA(input_explicit_x))
    @test DE_ref_1d ≈ DE_ref_ex
    @test DE_trn_1d ≈ DE_trn_ex
end

# A pattern without any mirror symmetry, so that it couples all 2D orders.
const _ASYMMETRIC_2D_PATTERN = [
    1.0 1.0 2.0 2.0 2.0 1.0
    1.0 1.0 2.0 2.0 1.0 1.0
    2.0 1.0 1.0 1.0 1.0 1.0
    2.0 2.0 1.0 1.0 1.0 1.0
]

@testset "2D energy conservation" begin
    # Truncation does not break energy conservation: the convolution matrices
    # of a lossless layer are Hermitian. This failed for every genuinely 2D
    # pattern before the branch fix in the homogeneous modes. The period avoids
    # Rayleigh anomalies, where an infinitesimal loss is added on purpose.
    stack = Stack(
        [Layer(1.0), Layer(_ASYMMETRIC_2D_PATTERN), Layer(1.5)],
        [Inf, 0.3, Inf], (1.05, 0.95)
    )
    for (ϕ, θ) in [(0.0, 0.0), (0.7, 0.4), (-2.3, 0.5), (2.0, 0.2)], N in (1, 3)
        result = RCWA(RCWAInput(Source(ϕ, θ, 0.6), stack, (N, N)))
        for pol in (:x, :y, :s, :p)
            DE_ref, DE_trn = diffraction_efficiencies(result; polarization = pol)
            @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 1e-10
        end
    end
end

@testset "Absorbing substrate" begin
    # The power flowing into an absorbing substrate equals the power that is not
    # reflected, provided the layers above are lossless. This checks the Poynting
    # vector for complex kz.
    n_Si = 4.15 + 0.044im
    stack = Stack(
        [Layer(1.0), Layer(_ASYMMETRIC_2D_PATTERN), Layer(n_Si)],
        [Inf, 0.3, Inf], (0.85, 0.7)
    )
    for (ϕ, θ) in [(0.0, 0.0), (-2.3, 0.5)]
        result = RCWA(RCWAInput(Source(ϕ, θ, 0.6), stack, (2, 2)))
        for pol in (:s, :p)
            DE_ref, DE_trn = diffraction_efficiencies(result; polarization = pol)
            @test sum(DE_ref) + sum(DE_trn) ≈ 1.0 atol = 1e-10
        end
    end
end

@testset "Diagonal grating" begin
    # A 1D grating along the diagonal of a square unit cell is a genuinely 2D
    # pattern, but it only couples the orders (m, m), which are exactly the
    # orders m of the equivalent 1D grating with period a/√2. At normal
    # incidence, X-polarized light is an equal mix of TE and TM light for the
    # diagonal grating, and TE and TM do not couple.
    Npix = 128
    a = 1.0
    profile = [k < Npix ÷ 2 ? 1.0 : 2.0 for k in 0:(Npix - 1)]
    # Pixel (ix, iy) has value profile[ix + iy]; row 1 is the +Y edge.
    pattern = [profile[mod(ix + (Npix - r), Npix) + 1] for r in 1:Npix, ix in 0:(Npix - 1)]

    N = 3
    source = Source(0.6)
    result_2d = RCWA(RCWAInput(
        source,
        Stack([Layer(1.0), Layer(pattern), Layer(1.5)], [Inf, 0.3, Inf], (a, a)),
        (N, N)
    ))
    result_1d = RCWA(RCWAInput(
        source,
        Stack([Layer(1.0), Layer(reshape(profile, 1, :)), Layer(1.5)],
              [Inf, 0.3, Inf], a / sqrt(2)),
        N
    ))

    DE_ref_2d, DE_trn_2d = diffraction_efficiencies(result_2d; polarization = :x)
    DE_ref_TM, DE_trn_TM = diffraction_efficiencies(result_1d; polarization = :x)
    DE_ref_TE, DE_trn_TE = diffraction_efficiencies(result_1d; polarization = :y)

    # The staircase pixels have slightly different Fourier coefficients than
    # the pixels of the 1D profile (by a factor of order 1 - (πm/Npix)²).
    for m in -N:N
        i = N + 1 + m
        @test DE_ref_2d[i, i] ≈ (DE_ref_TM[i] + DE_ref_TE[i]) / 2 atol = 1e-3
        @test DE_trn_2d[i, i] ≈ (DE_trn_TM[i] + DE_trn_TE[i]) / 2 atol = 1e-3
    end
    off_diagonal = sum(DE_ref_2d) + sum(DE_trn_2d) - sum(DE_ref_2d[i, i] + DE_trn_2d[i, i] for i in 1:(2N + 1))
    @test abs(off_diagonal) < 1e-12
end

# Test to end all tests: 2D regression against reference implementation.
# Used Python's grcwa as reference.
# Reference numbers were produced with the following Python script:
#=
import numpy as np
import grcwa

grcwa.set_backend("numpy")

# CasualRCWA treats pixels as piecewise-constant functions. grcwa takes DFT of
# point samples, where sample (i, j) of an Mx x My grid sits at (i/Mx, j/My) of
# the unit cell. So we sample the pattern on a grid S times finer than its
# pixels. Where a sample falls on a pixel we use the average of the adjacent
# pixels.
S = 400


def sample_grid(eps):
    # `eps` is in grcwa's orientation eps[ix, iy] (y upward); pixel p along an
    # axis with N pixels covers [p/N, (p+1)/N).
    Nx, Ny = eps.shape
    ix = np.arange(Nx * S)
    iy = np.arange(Ny * S)
    px, py = ix // S, iy // S
    qx = np.where(ix % S == 0, (px - 1) % Nx, px)  # Left neighbor on an edge.
    qy = np.where(iy % S == 0, (py - 1) % Ny, py)  # Lower neighbor on an edge.
    return 0.25 * (eps[np.ix_(px, py)] + eps[np.ix_(qx, py)]
                   + eps[np.ix_(px, qy)] + eps[np.ix_(qx, qy)])


# Patterns in CasualRCWA's convention: columns = X, rows = Y with row 1 = +Y.
# Entries are material codes.
PATTERN_A = np.array([
    [1, 1, 2, 2, 2, 1],
    [1, 1, 2, 2, 1, 1],
    [2, 1, 1, 1, 1, 1],
    [2, 2, 1, 1, 1, 1],
])
PATTERN_B = np.array([
    [1, 1, 1, 3, 3],
    [1, 2, 2, 3, 3],
    [1, 2, 2, 1, 1],
    [1, 1, 1, 1, 1],
    [3, 1, 1, 1, 2],
])
N_SI = 4.15 + 0.044j

# Layers between the half-spaces: (thickness, n) for a homogeneous layer, or
# (thickness, pattern, {code: n}) for a patterned layer.
CASES = {
    "normal_incidence": dict(
        wavelength=0.6, period=(0.85, 0.7), theta=0.0, phi=0.0,
        n_top=1.0, n_bottom=1.5,
        layers=[(0.3, PATTERN_A, {1: 1.0, 2: 2.0})],
    ),
    "conical_incidence": dict(
        wavelength=0.6, period=(0.85, 0.7), theta=30.0, phi=50.0,
        n_top=1.0, n_bottom=1.5,
        layers=[(0.3, PATTERN_A, {1: 1.0, 2: 2.0})],
    ),
    "lossy_negative_azimuth": dict(
        wavelength=0.532, period=(0.8, 0.7), theta=30.0, phi=-130.0,
        n_top=1.0, n_bottom=1.45,
        layers=[(0.12, PATTERN_B, {1: 1.0, 2: N_SI, 3: N_SI})],
    ),
    "multilayer_water_top": dict(
        wavelength=0.6, period=(0.8, 0.9), theta=20.0, phi=75.0,
        n_top=1.33, n_bottom=1.5,
        layers=[
            (0.2, PATTERN_A, {1: 1.33, 2: 1.8}),
            (0.1, 2.0),
            (0.15, PATTERN_B, {1: 1.5, 2: 2.2, 3: 1.0}),
        ],
    ),
}
N = 2  # Orders -N..N in both X and Y.


def transverse_e(obj, layer, a, b):
    # Transverse E-field Fourier components (as in grcwa's Solve_FieldFourier).
    tmp = obj.phi_list[layer] @ ((a - b) / obj.omega / obj.q_list[layer])
    fexy = obj.kp_list[layer] @ tmp
    return fexy[obj.nG:], -fexy[:obj.nG]


def simulate(case, pol):
    c = CASES[case]
    Lx, Ly = c["period"]
    obj = grcwa.obj((2 * N + 1) ** 2, [Lx, 0], [0, Ly], 1 / c["wavelength"],
                    np.deg2rad(c["theta"]), np.deg2rad(c["phi"]), verbose=0)
    # Zero-thickness half-spaces put all amplitude reference planes at the
    # outer interfaces, like in CasualRCWA.
    obj.Add_LayerUniform(0.0, c["n_top"] ** 2)
    grids = []
    for layer in c["layers"]:
        if len(layer) == 2:
            obj.Add_LayerUniform(layer[0], layer[1] ** 2)
        else:
            d, pattern, materials = layer
            n = np.vectorize(materials.get, otypes=[complex])(pattern)
            eps = np.flipud(n).T ** 2  # grcwa grid is eps[ix, iy], y upward
            grid = sample_grid(eps)
            obj.Add_LayerGrid(d, *grid.shape)
            grids.append(grid.flatten())
    obj.Add_LayerUniform(0.0, c["n_bottom"] ** 2)
    obj.Init_Setup(Gmethod=1)
    obj.GridLayer_geteps(np.concatenate(grids))
    if pol == "s":
        obj.MakeExcitationPlanewave(0, 0, 1, 0, order=0)
    else:
        obj.MakeExcitationPlanewave(1, 0, 0, 0, order=0)

    G = [tuple(g) for g in obj.G]
    assert sorted(G) == [(m, n) for m in range(-N, N + 1) for n in range(-N, N + 1)]
    zeroth = G.index((0, 0))

    # Normalize amplitudes to an incident field with transverse components
    # equal to CasualRCWA's Jones vector for :s or :p. grcwa's plane wave has
    # unit H-field, so alpha = 1/n_top.
    θ, ϕ = np.deg2rad(c["theta"]), np.deg2rad(c["phi"])
    jones = np.array([-np.sin(ϕ), np.cos(ϕ)]) if pol == "s" else \
        np.cos(θ) * np.array([np.cos(ϕ), np.sin(ϕ)])
    a0, b0 = obj.GetAmplitudes(0, 0.0)
    ex, ey = transverse_e(obj, 0, a0, 0 * b0)
    e_inc = np.array([ex[zeroth], ey[zeroth]])
    alpha = (e_inc @ np.conj(jones)) / (jones @ jones)
    assert np.allclose(e_inc, alpha * jones, atol=1e-14)
    assert np.isclose(alpha, 1 / c["n_top"], atol=1e-14)

    rx, ry = transverse_e(obj, 0, 0 * a0, b0)
    aN, bN = obj.GetAmplitudes(obj.Layer_N - 1, 0.0)
    tx, ty = transverse_e(obj, obj.Layer_N - 1, aN, 0 * bN)
    R, T = obj.RT_Solve(normalize=1, byorder=1)

    # Propagating orders in either half-space.
    kz2_top = c["n_top"] ** 2 * obj.omega ** 2 - obj.kx ** 2 - obj.ky ** 2
    kz2_bottom = c["n_bottom"] ** 2 * obj.omega ** 2 - obj.kx ** 2 - obj.ky ** 2
    assert np.min(np.abs(np.concatenate([kz2_top, kz2_bottom]))) > 0.05 * obj.omega ** 2
    rows = []
    for i in np.argsort([g[0] * 100 + g[1] for g in G]):
        if kz2_top[i] > 0 or kz2_bottom[i] > 0:
            rows.append((G[i], R[i], T[i], rx[i] / alpha, ry[i] / alpha,
                         tx[i] / alpha, ty[i] / alpha))
    return rows


def jl(z):
    return f"{z.real: .8f} {'+' if z.imag >= 0 else '-'} {abs(z.imag):.8f}im"


if __name__ == "__main__":
    for case in CASES:
        for pol in ("s", "p"):
            rows = simulate(case, pol)
            print(f'    ("{case}", :{pol}) => [')
            for (m, n), R, T, rx, ry, tx, ty in rows:
                print(f"        (({m:2d}, {n:2d}), {R:.8f}, {T:.8f}, "
                      f"{jl(rx)}, {jl(ry)}, {jl(tx)}, {jl(ty)}),")
            print("    ],")
=#

# The cases of the script above, in CasualRCWA's language.
const PATTERN_A = [
    1 1 2 2 2 1
    1 1 2 2 1 1
    2 1 1 1 1 1
    2 2 1 1 1 1
]
const PATTERN_B = [
    1 1 1 3 3
    1 2 2 3 3
    1 2 2 1 1
    1 1 1 1 1
    3 1 1 1 2
]
const N_SI = 4.15 + 0.044im

_patterned(pattern, materials) = Layer([materials[c] for c in pattern])

# Layers between the half-spaces as (thickness, layer).
const GRCWA_CASES = Dict(
    "normal_incidence" => (
        wavelength = 0.6, period = (0.85, 0.7), θ = 0.0, ϕ = 0.0,
        n_top = 1.0, n_bottom = 1.5,
        layers = [(0.3, _patterned(PATTERN_A, Dict(1 => 1.0, 2 => 2.0)))],
    ),
    "conical_incidence" => (
        wavelength = 0.6, period = (0.85, 0.7), θ = 30.0, ϕ = 50.0,
        n_top = 1.0, n_bottom = 1.5,
        layers = [(0.3, _patterned(PATTERN_A, Dict(1 => 1.0, 2 => 2.0)))],
    ),
    "lossy_negative_azimuth" => (
        wavelength = 0.532, period = (0.8, 0.7), θ = 30.0, ϕ = -130.0,
        n_top = 1.0, n_bottom = 1.45,
        layers = [(0.12, _patterned(PATTERN_B, Dict(1 => 1.0, 2 => N_SI, 3 => N_SI)))],
    ),
    "multilayer_water_top" => (
        wavelength = 0.6, period = (0.8, 0.9), θ = 20.0, ϕ = 75.0,
        n_top = 1.33, n_bottom = 1.5,
        layers = [
            (0.2, _patterned(PATTERN_A, Dict(1 => 1.33, 2 => 1.8))),
            (0.1, Layer(2.0)),
            (0.15, _patterned(PATTERN_B, Dict(1 => 1.5, 2 => 2.2, 3 => 1.0))),
        ],
    ),
)
const GRCWA_ORDER = 2

# Output of the script above: (case, polarization) => rows of
# ((m, n), R, T, r_x, r_y, t_x, t_y).
const GRCWA_REFERENCE = Dict(
    ("normal_incidence", :s) => [
        ((-2,  0), 0.00000000, 0.07134407, -0.01183887 + 0.00354924im, -0.03635123 - 0.03484560im,  0.00045971 + 0.00421106im,  0.36355575 - 0.09177889im),
        ((-1, -1), 0.00000000, 0.17458009,  0.03487561 + 0.08449156im,  0.00116323 - 0.04853102im, -0.08035805 - 0.09804771im,  0.08070957 + 0.32664205im),
        ((-1,  0), 0.00120757, 0.02273810, -0.01478735 - 0.00531762im,  0.03457761 + 0.00412528im,  0.05042245 - 0.08549682im, -0.02721034 + 0.06152223im),
        ((-1,  1), -0.00000000, 0.14463649,  0.02655820 + 0.12419890im, -0.04865285 + 0.07444693im, -0.08177735 - 0.08814355im, -0.10682233 - 0.28795739im),
        (( 0, -1), 0.00329368, 0.03378586, -0.03590110 - 0.00332197im, -0.03676028 - 0.00055463im, -0.04302314 - 0.07001165im,  0.05167799 - 0.10614134im),
        (( 0,  0), 0.04993159, 0.17413795,  0.00866558 + 0.01961824im, -0.20239055 - 0.09224796im,  0.00921009 + 0.03874921im, -0.05716241 - 0.33352376im),
        (( 0,  1), 0.00092605, 0.00696033,  0.02639280 + 0.01740406im,  0.01375344 - 0.00476068im, -0.05932207 - 0.01044160im,  0.03693318 + 0.00071040im),
        (( 1, -1), -0.00000000, 0.09126668, -0.13558660 - 0.04130515im, -0.05922697 - 0.07023138im,  0.14308379 - 0.07532524im,  0.22940814 - 0.00054774im),
        (( 1,  0), 0.00193358, 0.08653937,  0.00911089 + 0.00420032im, -0.03986588 + 0.03065756im, -0.02981945 + 0.04259434im, -0.23889974 - 0.06956832im),
        (( 1,  1), -0.00000000, 0.10409378, -0.07305225 + 0.00324439im,  0.04288156 + 0.02533719im,  0.12060697 - 0.01461597im, -0.22999464 - 0.11092447im),
        (( 2,  0), -0.00000000, 0.03262482, -0.00103231 - 0.00814472im, -0.00269876 + 0.01848657im, -0.00392436 - 0.00693240im, -0.13939050 - 0.21066418im),
    ],
    ("normal_incidence", :p) => [
        ((-2,  0), -0.00000000, 0.02374737,  0.01505806 + 0.00854034im, -0.01101807 - 0.01679175im,  0.05873614 + 0.04320519im,  0.01262344 - 0.01142990im),
        ((-1, -1), 0.00000000, 0.13883943,  0.00046997 - 0.06312068im,  0.00855691 + 0.06231442im,  0.10150808 + 0.30841584im, -0.06877294 - 0.11978033im),
        ((-1,  0), 0.00053895, 0.01821206,  0.01828592 + 0.00334721im,  0.00789339 - 0.00313035im, -0.04870352 + 0.03665274im,  0.06113472 - 0.07246016im),
        ((-1,  1), -0.00000000, 0.17475598, -0.07759693 + 0.10235560im, -0.01687348 + 0.10950620im, -0.14280005 - 0.33271614im, -0.10151106 - 0.10375280im),
        (( 0, -1), 0.00211525, 0.07122318, -0.05606287 - 0.02759302im, -0.00427149 - 0.00595033im,  0.13596297 - 0.18833164im,  0.00331694 - 0.05117051im),
        (( 0,  0), 0.04199684, 0.13626321, -0.18993248 - 0.07390888im,  0.00866558 + 0.01961824im, -0.05620903 - 0.29297497im, -0.00919738 + 0.04199718im),
        (( 0,  1), 0.00052533, 0.02855771, -0.01517255 - 0.01655520im,  0.00068415 + 0.01167612im,  0.11854653 + 0.04739257im, -0.06816783 + 0.00001420im),
        (( 1, -1), -0.00000000, 0.13869673, -0.07521541 - 0.10359661im, -0.10342244 - 0.06331520im,  0.31074080 - 0.09438121im,  0.13841087 - 0.04781656im),
        (( 1,  0), 0.00207207, 0.08286702, -0.00182915 + 0.03670548im,  0.00929476 + 0.01212161im, -0.19698754 - 0.07911843im, -0.01929565 + 0.06600264im),
        (( 1,  1), 0.00000000, 0.12367266,  0.07310628 + 0.05523224im, -0.06197277 - 0.01958955im, -0.28214116 - 0.08352952im,  0.11422715 - 0.05084444im),
        (( 2,  0), 0.00000000, 0.01591621, -0.01345328 - 0.02699284im, -0.02022444 + 0.01851957im, -0.01176652 - 0.05546407im, -0.05036834 - 0.02665894im),
    ],
    ("conical_incidence", :s) => [
        ((-2, -1), -0.00000000, 0.04120594, -0.01863157 + 0.02843248im,  0.01382305 - 0.10263098im,  0.01232313 - 0.12976976im,  0.02672530 + 0.09908778im),
        ((-2,  0), 0.00000000, 0.03313635, -0.05025833 + 0.03025768im, -0.07506506 + 0.05981614im, -0.08734565 - 0.07902222im, -0.05031739 - 0.04709381im),
        ((-1, -2), -0.00000000, 0.04843105, -0.04227199 + 0.04148016im, -0.01349455 - 0.01156654im,  0.00228179 - 0.26113248im, -0.00221229 + 0.06208217im),
        ((-1, -1), 0.00470191, 0.20429838, -0.03067484 + 0.02085271im,  0.05707352 - 0.01115437im,  0.02972893 - 0.20372928im, -0.02423375 + 0.29030580im),
        ((-1,  0), 0.00204803, 0.04729676, -0.03603556 - 0.00433902im,  0.01513807 - 0.00419587im,  0.06146007 - 0.10676680im,  0.00017482 + 0.10261056im),
        ((-1,  1), -0.00000000, 0.09848870, -0.04052090 - 0.04782035im, -0.09407513 - 0.02910504im,  0.24278654 + 0.11037025im, -0.00871922 - 0.05226723im),
        (( 0, -2), 0.00000000, 0.01356797,  0.01933650 - 0.01638923im, -0.00073977 - 0.01780385im,  0.02032926 + 0.07282132im, -0.04053717 + 0.03229535im),
        (( 0, -1), 0.00269719, 0.04236933,  0.04299336 + 0.02594189im,  0.00469322 + 0.00904100im, -0.10087725 + 0.02543263im,  0.10583400 - 0.02598623im),
        (( 0,  0), 0.06496375, 0.25514989,  0.19764562 + 0.02843166im, -0.15641952 - 0.02463820im,  0.06962622 + 0.30434127im, -0.08216982 - 0.22787415im),
        (( 0,  1), -0.00000000, 0.00689876,  0.05017267 - 0.08140702im, -0.03638797 + 0.02996680im, -0.08309896 + 0.01550284im,  0.00980373 - 0.00688242im),
        (( 1, -1), 0.00000000, 0.05417881, -0.03697207 + 0.03506027im, -0.02007530 - 0.02604076im, -0.06913356 + 0.00892447im,  0.14751774 + 0.02307331im),
        (( 1,  0), -0.00000000, 0.08056718, -0.04097796 - 0.04064008im,  0.10515122 + 0.02806235im,  0.10203753 - 0.10205556im, -0.15096302 + 0.14134875im),
    ],
    ("conical_incidence", :p) => [
        ((-2, -1), -0.00000000, 0.03874742, -0.00494317 + 0.05428621im,  0.05969004 - 0.06910489im, -0.03479127 + 0.04597634im,  0.09925354 + 0.10640546im),
        ((-2,  0), -0.00000000, 0.07763786,  0.00807663 - 0.00995145im, -0.12517505 - 0.08727558im,  0.04850061 + 0.05211182im,  0.24186724 - 0.02073919im),
        ((-1, -2), -0.00000000, 0.00703743, -0.02146480 - 0.04274131im, -0.01686200 - 0.01204975im,  0.00243730 + 0.07404362im, -0.00679817 + 0.00851971im),
        ((-1, -1), 0.00305606, 0.13798991, -0.00621043 - 0.02314382im,  0.04137884 - 0.01401967im, -0.05805902 + 0.09338112im, -0.07824295 + 0.23561999im),
        ((-1,  0), 0.00069495, 0.01595487, -0.00984715 + 0.01444820im,  0.01532802 + 0.00000868im, -0.04065845 - 0.06134960im,  0.03472213 + 0.04455548im),
        ((-1,  1), 0.00000000, 0.10607911,  0.08222774 + 0.14015211im,  0.01594442 - 0.01391817im, -0.27529963 - 0.06191835im, -0.15629326 - 0.04788877im),
        (( 0, -2), -0.00000000, 0.01573249,  0.03364594 + 0.00948229im, -0.00795128 + 0.03096064im, -0.04292676 + 0.03060672im, -0.06633855 + 0.00003970im),
        (( 0, -1), 0.00479694, 0.10017727, -0.05894073 - 0.02856434im, -0.02314812 + 0.00149281im,  0.13577186 - 0.15975352im,  0.06675523 - 0.11866258im),
        (( 0,  0), 0.03063473, 0.21200672, -0.09872276 - 0.03571484im, -0.10304613 - 0.03689406im, -0.05014179 - 0.21948741im, -0.06231127 - 0.24665915im),
        (( 0,  1), -0.00000000, 0.01215944, -0.02255245 - 0.01240262im,  0.06183902 + 0.04164198im,  0.08578602 - 0.01578780im, -0.03911587 + 0.03499771im),
        (( 1, -1), -0.00000000, 0.13557720, -0.05318680 - 0.02564548im, -0.16372007 - 0.02851996im,  0.17896252 - 0.12154894im,  0.23177958 - 0.04697706im),
        (( 1,  0), -0.00000000, 0.10171759, -0.02842747 + 0.00683344im,  0.08904672 + 0.00974544im, -0.08229261 + 0.10111446im, -0.03770612 + 0.17455382im),
    ],
    ("lossy_negative_azimuth", :s) => [
        ((-1,  0), -0.00000000, 0.04770958, -0.01572953 + 0.03981414im,  0.08819189 - 0.14808822im, -0.05139756 - 0.05185022im,  0.16857547 + 0.08600286im),
        ((-1,  1), -0.00000000, 0.09311812, -0.17982961 - 0.02839636im, -0.04692026 - 0.02887635im, -0.07702161 - 0.15752709im, -0.06144954 + 0.08817586im),
        (( 0, -1), -0.00000000, 0.06390618,  0.12756954 + 0.13042995im, -0.01255632 - 0.04631945im, -0.12325325 + 0.01270116im,  0.15464668 + 0.00239969im),
        (( 0,  0), 0.18738176, 0.01231002, -0.31505716 - 0.13101397im,  0.25700563 + 0.06737430im,  0.00410489 + 0.04028245im,  0.05687453 - 0.05143615im),
        (( 0,  1), 0.01666559, 0.11478250,  0.03579225 + 0.00808491im, -0.04086547 + 0.10419145im,  0.21352705 - 0.08031182im, -0.11679307 - 0.01413530im),
        (( 0,  2), 0.00000000, 0.09860990,  0.01712796 - 0.01907872im,  0.05889828 - 0.10864656im, -0.18977200 - 0.03900069im,  0.07293197 + 0.09471957im),
        (( 1, -1), -0.00000000, 0.11431485, -0.06429060 - 0.06097168im, -0.06810924 + 0.15329443im,  0.16930076 + 0.07174718im, -0.05361011 + 0.15997046im),
        (( 1,  0), 0.01374182, 0.00691952, -0.04304489 + 0.06617568im,  0.05222492 - 0.03745221im,  0.01860263 - 0.01320550im,  0.01366424 - 0.05942909im),
        (( 1,  1), 0.00514098, 0.02995763, -0.03507769 + 0.02551923im,  0.05380436 + 0.00986461im, -0.01512728 - 0.08820009im, -0.01828255 + 0.10321008im),
        (( 1,  2), 0.00000000, 0.04578839,  0.02727656 + 0.06593582im,  0.02763772 - 0.15424559im, -0.15427457 + 0.10033957im,  0.09218052 - 0.03681182im),
        (( 2,  0), -0.00000000, 0.02714820,  0.05271293 - 0.02480638im, -0.08367585 - 0.05810418im,  0.03732271 + 0.05349238im, -0.01998828 - 0.09388563im),
        (( 2,  1), -0.00000000, 0.01404816, -0.03384480 - 0.04137463im, -0.10050359 + 0.00834971im, -0.04856950 - 0.00391830im,  0.09947662 + 0.01085373im),
    ],
    ("lossy_negative_azimuth", :p) => [
        ((-1,  0), -0.00000000, 0.02348915,  0.02094408 - 0.05333807im, -0.12550630 - 0.10460922im,  0.03542829 + 0.02245631im, -0.06968047 - 0.11549335im),
        ((-1,  1), -0.00000000, 0.01680317,  0.16546406 - 0.09665208im,  0.09927675 - 0.25936797im,  0.05970328 - 0.01631275im,  0.09711973 + 0.01846126im),
        (( 0, -1), 0.00000000, 0.11387344,  0.12005945 - 0.06488830im,  0.16576921 - 0.02400582im,  0.11024245 + 0.11329489im,  0.13292173 + 0.07129943im),
        (( 0,  0), 0.18482549, 0.02573656,  0.13037134 + 0.18582520im,  0.15657177 + 0.25032567im,  0.04088000 + 0.00111781im, -0.03725137 - 0.11114474im),
        (( 0,  1), 0.03042751, 0.05660538, -0.02113888 + 0.02820859im, -0.03248873 + 0.15705314im, -0.18082563 + 0.03305378im, -0.02942115 + 0.01002597im),
        (( 0,  2), -0.00000000, 0.01798362,  0.11866271 + 0.19747067im,  0.06594679 + 0.02539842im,  0.08036652 - 0.00640364im,  0.07759387 - 0.01441253im),
        (( 1, -1), 0.00000000, 0.11975438,  0.23375652 - 0.12603439im,  0.05753311 - 0.09679966im, -0.13615794 - 0.03738994im, -0.11118506 + 0.16206566im),
        (( 1,  0), 0.06095525, 0.00935375, -0.19019223 - 0.04930317im,  0.03152751 - 0.11339428im,  0.00514481 - 0.00264918im,  0.05780486 - 0.04706742im),
        (( 1,  1), 0.01031800, 0.03732043,  0.05840925 + 0.00002322im, -0.07072544 + 0.04008997im, -0.01870006 + 0.09281299im, -0.06289655 - 0.10186291im),
        (( 1,  2), -0.00000000, 0.03403619,  0.05113330 - 0.04695493im,  0.15322952 - 0.08981541im,  0.12182999 - 0.08487089im,  0.03620959 + 0.04097294im),
        (( 2,  0), -0.00000000, 0.04611618, -0.01503271 + 0.02101722im,  0.03786804 + 0.10094056im, -0.08756414 - 0.08867427im, -0.11395163 - 0.08427601im),
        (( 2,  1), 0.00000000, 0.03898027,  0.10428437 - 0.04794998im,  0.04569689 + 0.10233571im,  0.02461502 + 0.05363865im,  0.14736540 - 0.02227327im),
    ],
    ("multilayer_water_top", :s) => [
        ((-2, -1), -0.00000000, 0.00404328, -0.00602058 - 0.01013083im,  0.01794973 - 0.04665824im, -0.02446039 - 0.00552282im, -0.04391326 + 0.00037457im),
        ((-2,  0), -0.00000000, 0.02145994, -0.00870823 + 0.00307274im,  0.03924641 + 0.03315926im, -0.03015524 - 0.08057944im, -0.03107356 - 0.05177163im),
        ((-1, -2), 0.00468174, 0.00137177, -0.04120285 - 0.06258476im,  0.02635776 + 0.01491826im, -0.00206136 + 0.00557264im,  0.03003899 + 0.00608746im),
        ((-1, -1), 0.01832941, 0.04449724, -0.08872602 - 0.07740469im,  0.05091534 + 0.02355480im,  0.16702270 + 0.04682927im, -0.08153292 - 0.00994807im),
        ((-1,  0), 0.00336931, 0.03236003, -0.00118289 - 0.03683015im, -0.00979922 + 0.03420513im, -0.05352768 + 0.13489217im, -0.03869720 - 0.04501066im),
        ((-1,  1), 0.02371408, 0.05137844,  0.17453259 - 0.12452979im,  0.04522714 - 0.06009861im, -0.24299270 - 0.07461414im, -0.07168616 - 0.04941907im),
        (( 0, -2), 0.00032873, 0.00201141,  0.01375982 + 0.00566166im, -0.00531246 - 0.00800749im,  0.03338547 + 0.02315745im,  0.00711446 - 0.01470360im),
        (( 0, -1), 0.00310074, 0.00268496,  0.03210054 + 0.03214711im, -0.02665046 + 0.01172950im,  0.00754003 - 0.04241782im, -0.01234306 + 0.01494806im),
        (( 0,  0), 0.08596930, 0.51502080,  0.27142135 + 0.07286079im, -0.08104363 - 0.02035568im, -0.32997329 - 0.56341450im,  0.07201555 + 0.13700407im),
        (( 0,  1), 0.02022462, 0.03167674,  0.08551875 + 0.15188783im,  0.01196259 - 0.04108962im, -0.18317634 + 0.07183765im,  0.01627052 - 0.01613020im),
        (( 1, -2), 0.00270891, 0.00637888,  0.05458620 + 0.00853118im,  0.03034479 + 0.02351512im, -0.00194073 - 0.04554205im, -0.05213712 - 0.00414247im),
        (( 1, -1), 0.00412709, 0.04781951, -0.02445632 + 0.04580701im,  0.01166077 + 0.02502422im,  0.11220254 + 0.10908109im, -0.03627088 - 0.08489901im),
        (( 1,  0), 0.00760870, 0.03795835, -0.00083875 - 0.07293177im,  0.02415226 + 0.04787084im, -0.15493304 + 0.00141858im,  0.10570839 + 0.01887203im),
        (( 1,  1), -0.00000000, 0.02717602, -0.10370407 + 0.04703016im,  0.11655756 - 0.05971399im,  0.19987787 - 0.02665823im, -0.14400962 - 0.00399720im),
    ],
    ("multilayer_water_top", :p) => [
        ((-2, -1), -0.00000000, 0.00350032, -0.02137003 + 0.00330273im, -0.01231521 - 0.07980948im,  0.00209198 + 0.02890055im, -0.04753876 - 0.01601253im),
        ((-2,  0), -0.00000000, 0.04464526, -0.06569918 - 0.03422301im, -0.13220888 - 0.00068637im,  0.04616413 + 0.05869576im,  0.26644954 + 0.21976334im),
        ((-1, -2), 0.00360158, 0.00233270,  0.00608087 - 0.00516667im,  0.03840649 + 0.02994780im, -0.00371633 - 0.00481338im,  0.01256114 - 0.03673093im),
        ((-1, -1), 0.02135521, 0.05229065, -0.01263336 + 0.01012923im,  0.12427808 + 0.08219529im,  0.00616714 - 0.02027581im, -0.18124443 - 0.11743125im),
        ((-1,  0), 0.00833121, 0.03590005,  0.00501056 - 0.01726435im,  0.02835344 + 0.08100392im,  0.02496541 - 0.02800978im,  0.00733660 - 0.17645728im),
        ((-1,  1), 0.01113097, 0.05720771, -0.10136033 + 0.10535568im, -0.07132457 + 0.07741115im,  0.11007862 + 0.10677196im,  0.18510785 + 0.05804798im),
        (( 0, -2), 0.00033012, 0.00361530,  0.00018084 - 0.00510125im, -0.01339308 - 0.00629536im, -0.00577689 - 0.02826399im,  0.03404413 - 0.02791016im),
        (( 0, -1), 0.01709156, 0.00164384,  0.00070187 - 0.02141051im, -0.12400050 + 0.00724721im,  0.01588539 + 0.00722844im, -0.02102604 + 0.02475202im),
        (( 0,  0), 0.09000771, 0.41296028, -0.03994989 - 0.01447326im, -0.27708695 - 0.03206721im,  0.08800385 + 0.15033133im,  0.26530033 + 0.47666864im),
        (( 0,  1), 0.00267026, 0.02633769, -0.05145999 - 0.00553864im,  0.01502239 - 0.02063787im,  0.04381946 - 0.03001763im,  0.11044741 - 0.02581234im),
        (( 1, -2), 0.00661672, 0.00220544, -0.03189844 + 0.05620351im, -0.02210581 + 0.09017028im,  0.00784568 + 0.00741435im, -0.01857452 - 0.02688161im),
        (( 1, -1), 0.00496479, 0.10793657,  0.02581451 - 0.04075676im, -0.04136580 - 0.00839612im, -0.08594088 - 0.03012506im, -0.13167317 - 0.29000170im),
        (( 1,  0), 0.00176838, 0.05729257, -0.03064634 + 0.01069931im,  0.03345650 - 0.00633616im,  0.01301091 - 0.04609479im,  0.21938445 + 0.05669086im),
        (( 1,  1), -0.00000000, 0.02426312, -0.01828617 + 0.06109521im, -0.01009741 - 0.06920710im, -0.00406047 - 0.04581582im, -0.09452377 + 0.05349525im),
    ],
)

@testset "2D gratings against grcwa" begin
    N = GRCWA_ORDER
    for case in sort(collect(keys(GRCWA_CASES))), pol in (:s, :p)
        @testset "$case, $pol" begin
            c = GRCWA_CASES[case]
            stack = Stack(
                [Layer(c.n_top); last.(c.layers); Layer(c.n_bottom)],
                [Inf; first.(c.layers); Inf],
                c.period
            )
            source = Source(deg2rad(c.ϕ), deg2rad(c.θ), c.wavelength)
            output = RCWA(RCWAInput(source, stack, (N, N)))

            DE_ref, DE_trn = diffraction_efficiencies(output; polarization = pol)
            r_x, r_y = reflection_coefficients(output; polarization = pol)
            t_x, t_y = transmission_coefficients(output; polarization = pol)

            rows = GRCWA_REFERENCE[(case, pol)]
            for ((m, n), R, T, rx, ry, tx, ty) in rows
                # Order (m, n) lives at index (N + 1 + m, N + 1 + n).
                i, j = N + 1 + m, N + 1 + n
                @test DE_ref[i, j] ≈ R atol = 1e-5
                @test DE_trn[i, j] ≈ T atol = 1e-5
                @test r_x[i, j] ≈ rx atol = 1e-5
                @test r_y[i, j] ≈ ry atol = 1e-5
                @test t_x[i, j] ≈ tx atol = 1e-5
                @test t_y[i, j] ≈ ty atol = 1e-5
            end

            # The remaining orders are evanescent and carry no power.
            propagating = Set(first.(rows))
            for m in -N:N, n in -N:N
                (m, n) in propagating && continue
                @test DE_ref[N + 1 + m, N + 1 + n] == 0
                @test DE_trn[N + 1 + m, N + 1 + n] == 0
            end
        end
    end
end
