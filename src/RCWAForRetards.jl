module RCWAForRetards

using FFTW
using LinearAlgebra
using ToeplitzMatrices

# Functionality to set up inputs.
export HomogeneousLayer
export IncomingWave
export Layer
export RCWASettings
export Stack

# Core function and results struct.
export RCWA
export RCWAResult

# Helper functions to interpret output.
export diffraction_efficiencies
export reflection_coefficients
export transmission_coefficients

include("BTTB.jl")
include("Layers.jl")
include("WaveVectorPreparation.jl")
include("LayerModes.jl")
include("ScatteringMatrices.jl")
include("RCWA.jl")

end # module RCWA
