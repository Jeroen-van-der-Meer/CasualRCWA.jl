module CasualRCWA

using FFTW
using LinearAlgebra
using ToeplitzMatrices

# Functionality to set up inputs.
export Layer
export Source
export Stack

# Core function.
export RCWA
export RCWAInput
export RCWAOutput

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
include("Shows.jl")

end # module RCWA
