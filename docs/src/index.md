# Introduction to RCWA

RCWA solves **Maxwell's equations**, the fundamental equations governing all
electromagnetic phenomena, for structures that are periodic in the plane and
layered in depth. A typical use case is a laser beam hitting a diffraction
grating: RCWA computes the complex amplitude (magnitude and phase) of every
reflected and transmitted diffraction order.

This document builds up the theory in stages. We start with Maxwell's
equations in a homogeneous medium, introduce the transfer-matrix method for
planar slabs (and its numerical pitfalls), and then show how RCWA generalises
the same ideas to periodic structures. A fully worked example of a 1D binary
grating ties everything together at the end.

## Electromagnetic waves and Maxwell's equations

Light is an electromagnetic wave: oscillating electric and magnetic fields that
propagate through space. In a medium with electric permittivity
$\varepsilon$ and magnetic permeability $\mu$, and in the absence of free
charges and currents, Maxwell's equations read

```math
\nabla \cdot (\varepsilon \mathbf{E}) = 0, \qquad
\nabla \cdot (\mu \mathbf{H}) = 0,
```

```math
\nabla \times \mathbf{E} = -\mu \frac{\partial \mathbf{H}}{\partial t}, \qquad
\nabla \times \mathbf{H} = \varepsilon \frac{\partial \mathbf{E}}{\partial t}.
```

Here $\mathbf{E}$ is the electric field and $\mathbf{H}$ is the magnetic
field. The material parameters $\varepsilon$ and $\mu$ can vary with
position; they describe how strongly the material responds to electric and
magnetic fields, respectively. Most optical materials are non-magnetic, meaning $\mu = 1$.

We work with **monochromatic** light at angular frequency $\omega$. Every
field then oscillates at the same frequency, and we may write

```math
\mathbf{E}(\mathbf{r}, t)
  = \mathbf{E}(\mathbf{r})\, e^{-i\omega t}, \qquad
\mathbf{H}(\mathbf{r}, t)
  = \mathbf{H}(\mathbf{r})\, e^{-i\omega t}.
```

When we substitute this into Maxwell's equations the common factor
$e^{-i\omega t}$ cancels, and every time derivative $\partial/\partial t$
is replaced by a factor $-i\omega$. The result is the **time-harmonic
Maxwell equations**, which involve only the spatial field profiles
$\mathbf{E}(\mathbf{r})$ and $\mathbf{H}(\mathbf{r})$:

```math
\nabla \times \mathbf{E} = i\omega\mu\,\mathbf{H}, \qquad
\nabla \times \mathbf{H} = -i\omega\varepsilon\,\mathbf{E}.
```

RCWA works entirely in this frequency-domain picture. The `Source` struct
specifies a wavelength $\lambda$, which sets the frequency via
$\omega = 2\pi c / \lambda$.

## Homogeneous media: reduction to the wave equation

In a **homogeneous** medium (uniform $\varepsilon$ and $\mu$), the two
curl equations above can be combined into a single equation for
$\mathbf{E}$ alone. Take the curl of the first equation:

```math
\nabla \times (\nabla \times \mathbf{E})
  = i\omega\mu \, \nabla \times \mathbf{H}
  = i\omega\mu \, (-i\omega\varepsilon\,\mathbf{E})
  = \omega^2 \varepsilon\mu \, \mathbf{E}.
```

Using the vector identity
$\nabla \times (\nabla \times \mathbf{E}) = -\nabla^2 \mathbf{E}$
(valid because $\nabla \cdot \mathbf{E} = 0$ in a source-free homogeneous
medium), this becomes the **vector wave equation**:

```math
\nabla^2 \mathbf{E} + \varepsilon\mu\,\omega^2\,\mathbf{E} = 0.
```

The simplest solutions are **plane waves**:

```math
\mathbf{E}(\mathbf{r})
  = \mathbf{E}_0 \, e^{i\mathbf{k}\cdot\mathbf{r}},
```

where $\mathbf{k} = (k_x, k_y, k_z)$ is the **wave vector**, which points in
the direction the wave travels.

### The refractive index

The vector wave equation relates the magnitude of the wave vector to the wavelength:

```math
|\mathbf{k}| = \frac{2\pi n}{\lambda}, \qquad
n = \sqrt{\varepsilon\mu},
```

where $n$ is the **refractive index** of the material. In general $n$ is a
**complex number**: $n = n' + in''$. The real part $n'$ determines the
wavelength inside the material (a higher $n'$ means a shorter wavelength, so
$|\mathbf{k}|$ is larger). The imaginary part $n''$ describes
**absorption**: substituting a complex $n$ into the plane wave gives

```math
e^{i k z} = e^{i (2\pi n / \lambda) z}
           = e^{-2\pi n'' z / \lambda} \, e^{i 2\pi n' z / \lambda}.
```

The first factor is an exponential decay: the wave loses amplitude as it
propagates, because the material absorbs energy. A transparent material like
glass has $n'' \approx 0$; a metal or semiconductor can have a sizeable
$n''$.

To give a sense of scale, here are approximate refractive indices of a few
familiar materials at visible wavelengths ($\lambda \approx 550\,\text{nm}$):

| Material  | $n'$ (real part) | $n''$ (imaginary part) | Character                     |
| :-------- | ---------------: | ---------------------: | :---------------------------- |
| Air       |              1.0 |            $\approx 0$ | Transparent, no refraction    |
| Water     |              1.3 |            $\approx 0$ | Transparent, mild bending     |
| Glass     |              1.5 |            $\approx 0$ | Transparent, moderate bending |
| Silicon   |              4.1 |                   0.04 | High index, slightly lossy    |
| Aluminium |              0.9 |                    6.5 | Metallic, highly absorbing    |
| Gold      |              0.3 |                    2.9 | Metallic, highly absorbing    |

### Evanescent waves

Recall the relation $|\mathbf{k}|^2 = (2\pi n/\lambda)^2$ between the wave vector and the wavelength. Writing out the components of this relation gives the
**dispersion relation**:

```math
k_x^2 + k_y^2 + k_z^2
  = \varepsilon\mu\left(\frac{2\pi}{\lambda}\right)^2.
```

So once the in-plane components $(k_x, k_y)$ are known, $k_z$ is fixed up
to a sign (which determines whether the wave is going up or down):

```math
k_z = \pm\sqrt{\varepsilon\mu\left(\frac{2\pi}{\lambda}\right)^2 - k_x^2 - k_y^2}.
```

When $k_x^2 + k_y^2$ exceeds $\varepsilon\mu\,(2\pi/\lambda)^2$, the
quantity under the square root becomes negative, making $k_z$ purely
imaginary. A plane wave with imaginary $k_z$ doesn't oscillate in the
z-direction; instead its amplitude decays exponentially:

```math
e^{ik_z z} = e^{-|k_z|\,z} \qquad \text{(for imaginary } k_z\text{)}.
```

Such a wave is called **evanescent**. Evanescent fields are real — they exist
as decaying electromagnetic fields near a surface — but they carry no
time-averaged power in the z-direction and die off within a fraction of a
wavelength. Only **propagating** waves (those with real $k_z$) appear as
actual beams at a distant detector.

To make this concrete, set $\lambda = 1$ and $n = 1$, and consider the $(x, z)$-plane for simplicity. The dispersion
relation becomes $k_x^2 + k_z^2 = (2\pi)^2$. A wave travelling to the right has $k_x = 2\pi$ and
$k_z = 0$:

```math
E(x) = e^{i\,2\pi\,x}.
```

This is a propagating wave: it oscillates with one full cycle per unit
length, which is exactly the wavelength. A wave at 45° has
$k_x = k_z = \sqrt{2}\,\pi$: it oscillates in both x and z but still
propagates.

But what if $k_x = 4\pi$ — twice the free-space wavenumber? Then
$k_z = \sqrt{(2\pi)^2 - (4\pi)^2} = i\,2\sqrt{3}\,\pi$, and the field
becomes

```math
E(x,z) = e^{i\,4\pi\,x}\;e^{-2\sqrt{3}\,\pi\,z}.
```

This wave oscillates rapidly in x but decays exponentially in z. It
cannot exist on its own in unbounded empty space: the solution decays in
one z-direction but blows up in the other, so it only makes physical sense
when a surface is present to cut off the divergent side. A diffraction
grating will turn out to allow for this situation: a periodic grating of periodicity $\Lambda$ allows you to add multiples of
$2\pi/\Lambda$ to $k_x$, and when $k_x$ is big enough, the
corresponding diffraction order becomes evanescent, decaying away from the
grating surface. We will see this happen explicitly in the grating example
later.

## The transfer-matrix method for layered media

Before tackling periodic structures, it is instructive to study the simpler
problem of light propagating through a stack of **homogeneous slabs**. This is
the domain of the **transfer-matrix method** (TMM).

### Setting up the problem

Consider a planar structure consisting of several layers stacked in the
z-direction. Each layer $j$ has refractive index $n_j$ and thickness
$d_j$. The top and bottom media are semi-infinite (half-spaces). A plane
wave with in-plane wave vector $(k_x, k_y)$ impinges on the stack from
above.

```
             incoming light
                   ↓

              medium 0 (n0)           semi-infinite
   ─────────────────────────────────
              layer 1 (n1)            d₁
   ─────────────────────────────────
              layer 2 (n2)            d₂
   ─────────────────────────────────
                   ⋮                   ⋮
   ─────────────────────────────────
              medium N (nN)           semi-infinite
```

Inside each homogeneous layer, the field is a superposition of two plane waves:
one propagating (or decaying) in the $+z$ direction with amplitude $a_j$,
and one in the $-z$ direction with amplitude $b_j$. The z-component of the
wave vector in layer $j$ is

```math
k_{z,j} = \pm \sqrt{n_j^2 \left(\frac{2\pi}{\lambda}\right)^2 - k_x^2 - k_y^2}.
```

The challenge is to relate the amplitudes in the topmost medium to those in
the bottommost medium, given the boundary conditions at every interface.

### Boundary conditions and interface matrices

At each interface, Maxwell's equations demand that the **tangential** components
of $\mathbf{E}$ and $\mathbf{H}$ are continuous. In the special case that $k_x = k_y = 0$ (in other words, the incoming light is at normal incidence),
these conditions
at the interface between layers $j$ and $j+1$ can be expressed as a relation

```math
\begin{pmatrix} a_{j+1} \\ b_{j+1} \end{pmatrix}
  = D_{j,j+1} \begin{pmatrix} a_j \\ b_j \end{pmatrix},
```

between the amplitudes $a_j$ and $a_{j+1}$ of the forward plane waves, and the amplitudes $b_j$ and $b_{j+1}$ of the backward plane waves. Here, $D_{j,j+1}$ is the **interface matrix**

```math
D_{j,j+1}
  = \frac{1}{2}
    \begin{pmatrix}
      1 + n_j/n_{j+1} & 1 - n_j/n_{j+1} \\
      1 - n_j/n_{j+1} & 1 + n_j/n_{j+1}
    \end{pmatrix}.
```

For oblique incidence, there's a more complicated equation, which moreover depends on the polarisation of the incoming light, but the principle underlying it is the same.

### Propagation matrices

The interface matrix relates amplitudes on the two sides of a boundary. But
to get from one boundary to the next, the wave must also travel through the
interior of a layer. During this traversal, the forward wave (of amplitude $a_j$) picks
up a phase $e^{ik_{z,j} d_j}$ and the backward wave (of amplitude $b_j$) picks up
$e^{-ik_{z,j} d_j}$. This is captured by the **propagation matrix**:

```math
P_j = \begin{pmatrix} e^{ik_{z,j}\,d_j} & 0 \\ 0 & e^{-ik_{z,j}\,d_j} \end{pmatrix}.
```

### The global transfer matrix

The so-called global transfer matrix of the stack is obtained by multiplying interface and
propagation matrices from top to bottom:

```math
T = D_{N-1,N} \, P_{N-1} \, D_{N-2,N-1} \, \cdots \, P_1 \, D_{0,1}.
```

The global transfer matrix $T$ relates the amplitudes in the bottom medium
to those in the top medium:

```math
\begin{pmatrix} a_N \\ b_N \end{pmatrix}
  = T \begin{pmatrix} a_0 \\ b_0 \end{pmatrix}.
```

Setting $b_N = 0$ (no upward wave in the substrate) and $a_0 = 1$ (unit
incident amplitude), the **reflection coefficient** and **transmission coefficient** are

```math
r = \frac{b_0}{a_0} = -\frac{T_{21}}{T_{22}}, \qquad
t = \frac{a_N}{a_0} = T_{11} - T_{12}\,\frac{T_{21}}{T_{22}}.
```

These are complex numbers. Their magnitude indicate the fraction of electric field amplitude that gets reflected resp. transmitted; their phase tells you how much the reflected and transmitted waves are shifted relative to the incident wave.

### Worked example: Fabry–Pérot etalon

A Fabry–Pérot etalon is a single slab of material (refractive index $n_f$;
thickness $d$) sandwiched between two identical media (refractive index
$n_0$).

```
             incoming light
                   ↓

              medium 0 (n0)           semi-infinite
   ─────────────────────────────────
                film (nf)             d
   ─────────────────────────────────
              medium 0 (n0)           semi-infinite
```

At normal incidence the full transfer matrix is
$T = D_{f,0}\,P_f\,D_{0,f}$. Carrying out the algebra gives

```math
T_{11} = \left(\cos\delta - \frac{i}{2}\left(\frac{n_0}{n_f} + \frac{n_f}{n_0}\right)\sin\delta\right) e^{-i \delta},
```

```math
T_{21} = -\frac{i}{2}\left(\frac{n_0}{n_f} - \frac{n_f}{n_0}\right) \sin\delta \; e^{i\delta},
```

where

```math
\delta = \frac{2\pi n_f d}{\lambda}.
```

This yields reflection and transmission coefficients

```math
r = \frac{r_{01}(1 - e^{2i\delta})}{1 - r_{01}^2\,e^{2i\delta}}, \qquad
t = \frac{(1 - r_{01}^2)\,e^{i\delta}}{1 - r_{01}^2\,e^{2i\delta}},
```

where $r_{01} = (n_0 - n_f)/(n_0 + n_f)$ is the Fresnel reflection
coefficient of a single interface. These are exactly the standard
**Fabry–Pérot (Airy) formulae**.

As an example, consider a glass slab ($n_f = 1.3$) in air at
$\lambda = 600\,\text{nm}$ and plot $|r|^2$ and $|t|^2$ as a function of
slab thickness:

```@example fabry_perot_plot
using CairoMakie

n0, nf = 1.0, 1.3
r01 = (n0 - nf) / (n0 + nf)
λ = 600.0 # nm

d = range(0, 1500, length=500)
δ = @. 2π * nf * d / λ
r = @. r01 * (1 - exp(2im * δ)) / (1 - r01^2 * exp(2im * δ))
t = @. (1 - r01^2) * exp(im * δ) / (1 - r01^2 * exp(2im * δ))

fig = Figure(size=(600, 350))
ax = Axis(fig[1, 1], xlabel="Slab thickness d (nm)", ylabel="Power fraction",
    title="Fabry–Pérot etalon: n = $nf in air, λ = $(Int(λ)) nm")
lines!(ax, d, abs2.(r), label="|r|²")
lines!(ax, d, abs2.(t), label="|t|²")
axislegend(ax, position=:rt)
fig
```

The reflectance oscillates between zero (at the resonances $d = m\lambda/(2n_f)$,
where the round-trip phase is a multiple of $2\pi$) and a maximum of about
6.6%. The transmittance does the opposite, reaching unity at every resonance.
Because the index contrast is modest ($r_{01} \approx -0.13$), the oscillations
are gentle; a higher-contrast film would show sharper dips and peaks.

### Numerical instability of transfer matrices
