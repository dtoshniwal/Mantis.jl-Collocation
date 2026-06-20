# Mantis

[![docs:stable](https://img.shields.io/badge/docs-stable-purple)](https://mantisfem.github.io/Mantis.jl/stable/)
[![docs:dev](https://img.shields.io/badge/docs-dev-purple)](https://mantisfem.github.io/Mantis.jl/dev/)
[![Build Status](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIlts.yml/badge.svg?branch=main)](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIlts.yml?query=branch%3Amain)
[![Build Status](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIv1withcov.yml/badge.svg?branch=main)](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIv1withcov.yml?query=branch%3Amain)
[![Build Status](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIpre.yml/badge.svg?branch=main)](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIpre.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/MantisFEM/Mantis.jl/graph/badge.svg?token=ZWA3YV3IB6)](https://codecov.io/gh/MantisFEM/Mantis.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

## Introduction

Welcome to the documentation for `Mantis`, a Julia package for high-order
structure-preserving finite element methods.

This package is designed based on the Finite Element Exterior Calculus (FEEC) framework,
which provides a rigorous foundation for designing structure-preserving discretisations for
PDEs, e.g., those arising in electromagnetism, fluid flows, and elasticity. Such
discretisations require finite element spaces which discretize the Hilbert complexes
associated to the PDEs, such as the de Rham complex for Maxwell's equations. `Mantis`
provides users with a flexible environment where they can implement FEEC using the natural
language of Exterior Calculus, allowing them to discretize PDEs using spaces of arbitrary
regularities. Some examples of supported finite element spaces are piecewise-polynomial
spaces, non-polynomial spaces (e.g., trigonometric, exponential, Tchebycheffian B-splines),
and adaptively-refinable spaces (e.g., hierarchical B-splines).

`Mantis` is free, open-source, and available under the
[EUPL licence](https://github.com/MantisFEM/Mantis.jl/blob/main/LICENSE).


## Installation

`Mantis` needs [Julia](https://julialang.org/) 1.10 or newer. If you do not already have
Julia, the easiest way to install it is with
[`juliaup`](https://github.com/JuliaLang/juliaup):

- macOS or Linux: `curl -fsSL https://install.julialang.org | sh`
- Windows: `winget install julia -s msstore`

`Mantis` is not in the Julia package registry, and the collocation features are on the
`feat/collocation` branch of this fork, so install it from there directly.

To use it from your own Julia environment, start Julia (by running `julia`) and run:

```julia
using Pkg
Pkg.add(url="https://github.com/dtoshniwal/Mantis.jl-Collocation", rev="feat/collocation")
```

You can then load it at any time with `using Mantis`.

To develop the package, or to run the examples and tests, clone the `feat/collocation` branch
and install its dependencies instead:

```bash
git clone -b feat/collocation https://github.com/dtoshniwal/Mantis.jl-Collocation
cd Mantis.jl-Collocation
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The `--project=.` flag tells Julia to use the package's own environment. Run the test suite
with:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```


## Building the documentation locally

The documentation is built with [Documenter](https://documenter.juliadocs.org/) and
[DocumenterVitepress](https://luxdl.github.io/DocumenterVitepress.jl/), which renders the site
with [VitePress](https://vitepress.dev/). You therefore need both Julia and
[Node.js](https://nodejs.org/), which provides the `npm` command.

Starting from a clone of the repository (see above), run the following from its root:

```bash
# 1. Install the JavaScript dependencies that VitePress uses.
cd docs
npm install
cd ..

# 2. Install the Julia dependencies for the docs. Mantis itself is picked up from the
#    repository, so there is nothing else to set up.
julia --project=docs -e 'using Pkg; Pkg.instantiate()'

# 3. Run the examples and build the site into docs/build.
julia --project=docs docs/make.jl
```

To preview the result in a browser with live reloading, run:

```bash
cd docs
npm run docs:dev
```

and open the local address it prints (usually http://localhost:5173).


## Authors
The `Mantis` package was created by
- Diogo C. Cabanas,
- Joey Dekker,
- Artur Palha,
- Deepesh Toshniwal,
  
from TU Delft's Institute of Applied Mathematics (DIAM).
