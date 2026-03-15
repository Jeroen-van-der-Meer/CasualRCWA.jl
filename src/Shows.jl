_fmt(z::ComplexF64) = imag(z) == 0 ? string(real(z)) : string(z)
_fmt_angle(rad::Float64) = string(round(rad2deg(rad); digits = 2))
_fmt_complex(n::ComplexF64) = string(round(n; digits = 2))
_is_nonmagnetic(layer::Layer) = all(layer.mu .== 1)

function _material_str(layer::Layer)
    if _is_homogeneous(layer)
        ε = first(layer.eps)
        μ = first(layer.mu)
        if μ == 1.0
            return "n = $(_fmt_complex(sqrt(ε)))"
        else
            return "ε = $(_fmt_complex(ε)), μ = $(_fmt_complex(μ))"
        end
    else
        Nx, Ny = size(layer.eps)
        return "$Nx × $Ny"
    end
end

# Resample real(ε) to a 2D grid of shade characters (X horizontal, Y vertical).
# Returns a vector of strings, top row first (+Y at top).
function _shade_map(layer::Layer, width::Int, height::Int,
                    eps_lo::Float64, eps_hi::Float64)
    eps_real = real.(layer.eps)  # (Nx, Ny), dim1=X, dim2=Y
    Nx, Ny = size(eps_real)
    span = eps_hi - eps_lo
    rows = Vector{String}(undef, height)
    for r in 1:height
        # Row 1 = top = high Y
        iy = clamp(floor(Int, (height - r + 0.5) / height * Ny) + 1, 1, Ny)
        chars = Vector{Char}(undef, width)
        for j in 1:width
            ix = clamp(floor(Int, (j - 0.5) / width * Nx) + 1, 1, Nx)
            t = span == 0 ? 0.5 : (eps_real[ix, iy] - eps_lo) / span
            chars[j] = _shade(t)
        end
        rows[r] = String(chars)
    end
    return rows
end

# FIXME: use _is_homogeneous_in_x and _is_homogeneous_in_y to deal with Y-gratings
function Base.show(io::IO, ::MIME"text/plain", layer::Layer)
    eps_lo, eps_hi = extrema(real.(layer.eps))
    if _is_homogeneous(layer)
        bar = _shade_bar(layer, 16, eps_lo, eps_hi)
        print(io, "Layer: ", bar, "  ", _material_str(layer))
    else
        Nx, Ny = size(layer.eps)
        width = 16
        # Correct for ~2:1 character aspect ratio in terminals.
        height = clamp(round(Int, width * Ny / Nx / 2), 1, 10)
        rows = _shade_map(layer, width, height, eps_lo, eps_hi)
        println(io, "Layer: ", _material_str(layer))
        for (i, row) in enumerate(rows)
            print(io, "  ", row)
            i < height && println(io)
        end
    end
end

function Base.show(io::IO, ::MIME"text/plain", source::Source)
    ϕ = _fmt_angle(source.azimuthalAngle)
    θ = _fmt_angle(source.elevationAngle)
    λ = source.wavelength
    print(io, "Source: λ = $λ, ϕ = $(ϕ)°, θ = $(θ)°")
end

# Shade bar visualization

const _SHADE_CHARS = ('·', '░', '▒', '▓', '█')

function _shade(t::Float64)
    i = clamp(round(Int, t * (length(_SHADE_CHARS) - 1)) + 1, 1, length(_SHADE_CHARS))
    return _SHADE_CHARS[i]
end

# Return a 1D profile of real(ε) averaged over Y, resampled to `width` pixels.
function _eps_profile(layer::Layer, width::Int)
    eps_real = real.(layer.eps)  # dims: (Nx, Ny)
    col_avg = vec(sum(eps_real; dims = 2) ./ size(eps_real, 2))
    Nx = length(col_avg)
    return [col_avg[clamp(floor(Int, (j - 0.5) / width * Nx) + 1, 1, Nx)]
            for j in 1:width]
end

function _shade_bar(layer::Layer, width::Int, eps_lo::Float64, eps_hi::Float64)
    profile = _eps_profile(layer, width)
    if any(isnan, profile)
        return repeat("?", width)
    end
    span = eps_hi - eps_lo
    if span == 0
        return repeat(string(_shade(0.5)), width)
    end
    return join(_shade.((profile .- eps_lo) ./ span))
end

function _show_stack(io::IO, stack::Stack; indent::String = "  ")
    n = length(stack.layers)

    # Global ε range for consistent shading.
    eps_lo, eps_hi = Inf, -Inf
    for layer in stack.layers
        if layer isa Layer
            lo, hi = extrema(real.(layer.eps))
            eps_lo = min(eps_lo, lo)
            eps_hi = max(eps_hi, hi)
        end
    end

    # Pre-compute thickness and material strings for alignment.
    thickness_strs = [isinf(d) ? "half-space" : "d = $d"
                      for d in stack.thicknesses]
    material_strs = [_material_str(layer) for layer in stack.layers]
    max_thickness_width = maximum(length, thickness_strs)

    bar_width = 16
    for (i, (layer, d)) in enumerate(zip(stack.layers, stack.thicknesses))
        bar = _shade_bar(layer, bar_width, eps_lo, eps_hi)
        t_str = rpad(thickness_strs[i], max_thickness_width)
        print(io, indent, bar, "  ", t_str, "  ", material_strs[i])
        i < n && println(io)
    end
end

function Base.show(io::IO, ::MIME"text/plain", stack::Stack)
    n = length(stack.layers)
    Lx, Ly = stack.period
    println(io, "Stack: $n layers, period = ($Lx, $Ly)")
    _show_stack(io, stack)
end

function Base.show(io::IO, ::MIME"text/plain", input::RCWAInput)
    N, M = input.order
    P, Q = 2N + 1, 2M + 1
    n = length(input.stack.layers)
    Lx, Ly = input.stack.period
    println(io, "RCWAInput:")
    println(io, "  ", sprint(show, MIME"text/plain"(), input.source))
    println(io, "  Stack: $n layers, period = ($Lx, $Ly)")
    _show_stack(io, input.stack; indent = "    ")
    println(io)
    print(io, "  Order: ($N, $M) → $P × $Q = $(P * Q) harmonics")
end

function Base.show(io::IO, ::MIME"text/plain", output::RCWAOutput)
    N, M = output.input.order
    P, Q = 2N + 1, 2M + 1
    n_layers = length(output.input.stack.layers)
    λ = output.input.source.wavelength
    println(io, "RCWAOutput:")
    println(io, "  λ = $λ, order = ($N, $M) → $P × $Q = $(P * Q) harmonics")
    print(io, "  $n_layers layers, S-matrix size = $(size(output.scatteringMatrix.S11, 1))")
end
