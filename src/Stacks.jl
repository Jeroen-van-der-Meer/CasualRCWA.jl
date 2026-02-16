# FIXME: Look at stack after treating the layers.
struct Stack
    topMedium::AbstractLayer
    layers::Vector{<:AbstractLayer}
    bottomMedium::AbstractLayer
    thicknesses::Vector{Float64}
    period::Tuple{Float64, Float64}

    function Stack(
        top_medium::AbstractLayer,
        layers::AbstractVector{<:AbstractLayer},
        bottom_medium::AbstractLayer,
        thicknesses::AbstractVector{<:Real},
        period::Tuple{<:Real, <:Real}
    )
        @assert is_homogeneous(top_medium)
        @assert length(layers) == length(thicknesses)
        @assert is_homogeneous(bottom_medium)
        @assert (period[1] > 0) && (period[2] > 0)
        return new(top_medium, layers, bottom_medium, thicknesses, period)
    end
end

struct RcwaSettings
    incomingWave::IncomingWave
    stack::Stack
    numberOfHarmonics::Tuple{Int64, Int64}
end
