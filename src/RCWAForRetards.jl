module RCWAForRetards

using FFTW
using LinearAlgebra
using ToeplitzMatrices

export BTTB

export convolve
export ConvolvedLayer
export EmptyLayer
export HomogeneousLayer
export Layer

export IncomingWave
export PreparedWaveVectors
export prepare_wave_vectors

export compute_modes
export LayerModes

export compute_flat_scattering_matrix
export compute_global_scattering_matrix
export compute_scattering_matrix
export compute_symmetric_scattering_matrix
export ScatteringMatrix
export SymmetricScatteringMatrix
export star_product

include("BTTB.jl")
include("Layers.jl")
include("WaveVectorPreparation.jl")
include("LayerModes.jl")
include("ScatteringMatrices.jl")
include("Stacks.jl")

end # module RCWA
