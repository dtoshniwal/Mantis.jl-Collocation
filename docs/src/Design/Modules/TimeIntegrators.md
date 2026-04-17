```@meta
CurrentModule = Mantis.TimeIntegrators
```
# TimeIntegrators

The time integration module implemented in `Mantis` is based on the framework developed by [Vos2011](@cite).
This framework allows for an easy implementation of a variety of explicit, implicit, and implicit-explicit (IMEX) time stepping schemes, and is based on the concept of general linear methods (see, for example, [Butcher2006](@cite)).
These methods are applicable to both ODEs and PDEs, so that both are available in `Mantis`. 

General linear methods can be characterised as follows [Vos2011](@cite). 
Consider the initial value problem as ODE
```math
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}), \quad \mathbf{y}(t_0) = \mathbf{y}_0\;,
```
where ``\mathbf{f}: \mathbb{R}^N \to \mathbb{R}^N``. 
The ``n``-th step of the GLM comprised of ``r`` steps and ``s`` stages is then formulatied as
```math
\begin{align}
    \mathbf{Y}_i &= \Delta t \sum_{j=1}^{s} a_{ij} \mathbf{F}_j + \sum_{j=1}^{r} u_{ij} \mathbf{y}_j^{n-1}, \quad 1 \leq i \leq s\;, \\
    \mathbf{y}_i^n &= \Delta t \sum_{j=1}^{s} b_{ij} \mathbf{F}_j + \sum_{j=1}^{r} v_{ij} \mathbf{y}_j^{n-1}, \quad 1 \leq i \leq r\;,
\end{align}
```
where ``\mathbf{Y}_i`` are called the stage values and ``\mathbf{F}_i`` are called the stage derivatives.
These two quantities are related by the differential equation
```math
\mathbf{F}_i = \mathbf{f}(\mathbf{Y}_i)\;.
```

## What are time integrators in `Mantis`?
The top-level type within the `TimeIntegrators` module is the `AbstractTimeIntegrator{num_stages, num_steps}` type. 
```@docs
AbstractTimeIntegrator
```

The `AbstractTimeIntegrator{num_stages, num_steps}` type has three concrete subtypes, each 
representing a specific class of time integrators.
```@docs
Explicit
Implicit
IMEX
```

## Pre-implemented schemes
`Mantis` provides a few pre-implemented schemes for convience. 
You can, of course, always add a new scheme without loss of performance.
See the section on [adding your own scheme](@ref TimeIntegratorsAddYourOwn) for more information.

## [Adding your own scheme](@id TimeIntegratorsAddYourOwn)

## All docstrings from Mantis.TimeIntegrators
```@docs
TimeIntegrationSolution
define_picard_solver_ode
mapIMEXMultiStageToScheme
mapButcherTableauToScheme
define_newton_solver_ode
timeIntegrate!
timeIntegrate_!
TimeLevels
define_fixed_point_relaxation_ode
shift_steps!
initializeScheme
TimeIntegrationOperators
define_explicit_ode
define_implicit_ode
define_imex_ode
define_implicit_linear
timeIntegrate
calc_step_derivatives!
mapMultiStepToScheme
```
