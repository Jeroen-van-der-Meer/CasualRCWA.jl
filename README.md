# CasualRCWA.jl

[![Build Status](https://github.com/Jeroen-van-der-Meer/CasualRCWA.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/Jeroen-van-der-Meer/CasualRCWA.jl/actions/workflows/ci.yml?query=branch%3Amaster)
[![Docs](https://img.shields.io/badge/Docs-stable-blue.svg)](https://Jeroen-van-der-Meer.github.io/CasualRCWA.jl/stable/)

A straightforward Julia implementation of Rigorous Coupled-Wave Analysis (RCWA)
for simulating diffraction from periodic structures.

## Example: 1D Silicon Grating

Simulate a binary silicon grating (period 1 um, 50% duty cycle, 500 nm thick) at
600 nm wavelength under normal incidence.

```julia
using CasualRCWA

# Normal incidence by default; wavelength 600 nm
source = Source(0.6);

# Refractive indices
n_air   = 1.0;
n_glass = 1.5;

# Layer stack: air; grating; glass
air     = Layer(n_air);
grating = Layer([n_air n_glass n_glass n_air]);
glass   = Layer(n_glass);

stack = Stack(
    [air, grating, glass], # layers
    [Inf, 0.5, Inf], # thicknesses in um
    1 # period in um
);

# Simulate up to 5th diffraction order
order = 5;

input = RCWAInput(source, stack, order)
```

```
RCWAInput:
  Source: λ = 0.6, ϕ = 0.0°, θ = 0.0°
  Stack: 3 layers, period = (1.0, 1.0)
    ················  half-space  n = 1.0 + 0.0im
    ····████████····  d = 0.5     4 × 1
    ████████████████  half-space  n = 1.5 + 0.0im
  Order: (5, 0) → 11 × 1 = 11 harmonics
```

To simulate the results, call the `RCWA` function, which accepts an `RCWAInput`
and returns an `RCWAOutput`.

```julia
output = RCWA(input)
```

```
RCWAOutput:
  λ = 0.6, order = (5, 0) → 11 × 1 = 11 harmonics
  3 layers, S-matrix size = 22
```

The `RCWAOutput` struct notably contains the scattering matrices and
accompanying wave vectors. Some helper functions can be used to directly extract
diffraction efficiencies:

```julia
# Diffraction efficiencies of X-polarized light
DE_ref, DE_trn = diffraction_efficiencies(output; polarization = :x);
println("Reflection: ",   round.(DE_ref; digits = 2))
println("Transmission: ", round.(DE_trn; digits = 2))
```

```
Reflection: [-0.0; -0.0; -0.0; -0.0; 0.01; 0.01; 0.01; -0.0; -0.0; -0.0; -0.0;;]
Transmission: [0.0; 0.0; 0.0; 0.03; 0.33; 0.25; 0.33; 0.03; 0.0; 0.0; 0.0;;]
```
