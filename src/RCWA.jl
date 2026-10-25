"""
    RCWA(input::RCWAInput; full_scattering_matrix = false) -> RCWAOutput

Run a full RCWA simulation and return an `RCWAOutput`.

# Arguments

- `input::RCWAInput`: Simulation parameters including the incoming wave, layer
  stack, and number of harmonics.
- `full_scattering_matrix::Bool = false`: By default, only the blocks `S11` and
  `S21` of the global scattering matrix are computed. These describe reflection
  and transmission of light incident from the top, which is all that
  `reflection_coefficients`, `transmission_coefficients` and
  `diffraction_efficiencies` need. Set this to `true` to also compute the blocks
  `S12` and `S22` for light incident from the bottom, at about 1.7 times the
  cost of assembling the scattering matrix.
"""
function RCWA(input::RCWAInput; full_scattering_matrix::Bool = false)
    wavelength = input.source.wavelength
    PQ = (2 * input.order[1] + 1, 2 * input.order[2] + 1)

    # Prepare wave vectors
    top_medium = first(input.stack.layers)
    bottom_medium = last(input.stack.layers)
    wave_vectors = prepare_wave_vectors(
        input.source, top_medium, bottom_medium,
        input.stack.period, PQ
    )

    # Convolve layers and compute eigenmodes. A layer that appears several
    # times in the stack is only solved once.
    empty_modes = compute_modes(convolve(Layer(1.0), PQ), wave_vectors)
    cache = IdDict{AbstractLayer, LayerModes}()
    modes = [
        get!(() -> compute_modes(convolve(l, PQ), wave_vectors), cache, l)
        for l in input.stack.layers
    ]

    # Global scattering matrix
    Sg = compute_global_scattering_matrix(
        modes, empty_modes, wavelength, input.stack.thicknesses;
        full = full_scattering_matrix
    )

    return RCWAOutput(input, wave_vectors, modes, Sg)
end
