"""
    RCWA(input::RCWAInput) -> RCWAOutput

Run a full RCWA simulation and return an `RCWAOutput`.

# Arguments

- `input::RCWAInput`: Simulation parameters including the incoming wave, layer
  stack, and number of harmonics.
"""
function RCWA(input::RCWAInput)
    wavelength = input.source.wavelength
    PQ = (2 * input.order[1] + 1, 2 * input.order[2] + 1)

    # Prepare wave vectors
    top_medium = first(input.stack.layers)
    bottom_medium = last(input.stack.layers)
    wave_vectors = prepare_wave_vectors(
        input.source, top_medium, bottom_medium,
        input.stack.period, PQ
    )

    # Convolve layers and compute eigenmodes
    empty_modes = compute_modes(convolve(Layer(1.0), PQ), wave_vectors)
    modes = [compute_modes(convolve(l, PQ), wave_vectors) for l in input.stack.layers]

    # Global scattering matrix
    Sg = compute_global_scattering_matrix(
        modes, empty_modes, wavelength, input.stack.thicknesses
    )

    return RCWAOutput(input, wave_vectors, modes, Sg)
end
