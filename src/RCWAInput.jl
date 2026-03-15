"""
    struct Stack

A stack of layers with associated thicknesses. The first and last layers are
semi-infinite half-spaces (thickness = `Inf`) and must be homogeneous.

# Properties

- `layers::Vector{<:AbstractLayer}`: All layers, including the top and bottom
  half-spaces as the first and last elements.
- `thicknesses::Vector{Float64}`: Thickness of each layer. The first and last
  elements must be `Inf`.
- `period::Tuple{Float64, Float64}`: Unit cell size in (X, Y).
"""
struct Stack
    layers::Vector{<:AbstractLayer}
    thicknesses::Vector{Float64}
    period::Tuple{Float64, Float64}

    function Stack(
        layers::AbstractVector{<:AbstractLayer},
        thicknesses::AbstractVector{<:Real},
        period::Tuple{<:Real, <:Real}
    )
        @assert length(layers) >= 2
        @assert length(layers) == length(thicknesses)
        @assert _is_homogeneous(first(layers))
        @assert first(thicknesses) == Inf
        @assert _is_homogeneous(last(layers))
        @assert last(thicknesses) == Inf
        @assert (period[1] > 0) && (period[2] > 0)
        return new(layers, thicknesses, period)
    end
end

# 0D convenience: all layers homogeneous, period is irrelevant.
function Stack(
    layers::AbstractVector{<:AbstractLayer},
    thicknesses::AbstractVector{<:Real}
)
    if all(_is_homogeneous(l) for l in layers)
        return Stack(layers, thicknesses, (1.0, 1.0))
    else
        error("All layers must be homogeneous when no period is given")
    end
end

# 1D convenience: single scalar period, direction inferred from layers.
function Stack(
    layers::AbstractVector{<:AbstractLayer},
    thicknesses::AbstractVector{<:Real},
    period::Real
)
    if all(_is_homogeneous_in_y(l) for l in layers)
        return Stack(layers, thicknesses, (period, 1.0))
    elseif all(_is_homogeneous_in_x(l) for l in layers)
        return Stack(layers, thicknesses, (1.0, period))
    else
        error("Layers must vary in exactly one direction for a scalar period")
    end
end

"""
    struct RCWAInput

All parameters needed to run an RCWA simulation.

# Properties

- `source::Source`: Incoming wave specification (angles, wavelength).
- `stack::Stack`: Layer stack including top and bottom half-spaces.
- `order::Tuple{Int64, Int64}`: Maximum harmonic order (N, M). The simulation
  includes orders -N..N and -M..M, for a total of (2N+1)(2M+1) harmonics.
"""
struct RCWAInput
    source::Source
    stack::Stack
    order::Tuple{Int64, Int64}
end

# 0D convenience: no harmonics needed for a fully homogeneous stack.
function RCWAInput(source::Source, stack::Stack)
    if all(_is_homogeneous(l) for l in stack.layers)
        return RCWAInput(source, stack, (0, 0))
    else
        error("All layers must be homogeneous when no orders are given")
    end
end

# 1D convenience: single order, direction inferred from layers.
function RCWAInput(source::Source, stack::Stack, order::Integer)
    if all(_is_homogeneous_in_y(l) for l in stack.layers)
        return RCWAInput(source, stack, (order, 0))
    elseif all(_is_homogeneous_in_x(l) for l in stack.layers)
        return RCWAInput(source, stack, (0, order))
    else
        error("Single order requires layers that vary in at most one direction")
    end
end
