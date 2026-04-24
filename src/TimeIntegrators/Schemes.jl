"""
mapButcherTableauToScheme::(Matrix{Float64}, Array{Float64}, Array{Float64}) -> AbstractTimeIntegrator \\

A::Matrix{Float64} is a square matrix of size sxs, where num_stages is the number of stages of the scheme \\
B::Array{Float64} is a vector of size num_stages \\
C::Array{Float64} is a vector of size num_stages

Implementation from article:
Vos, Peter & Eskilsson, Claes & Bolis, Alessandro & Chun, Sehun & Kirby, Robert & Sherwin, Spencer. (2011).
A generic framework for time-stepping partial differential equations (PDEs):
General linear methods, object-oriented implementation and application to fluid problems.
International Journal of Computational Fluid Dynamics. 25. 107-125. 10.1080/10618562.2011.575368.

Specficly Appendix A.1
"""
function mapButcherTableauToScheme(
    A::SMatrix{num_stages, num_stages, T},
    B::SVector{num_stages, T},
    C::SVector{num_stages, T},
    order::Int,
) where {num_stages, T}
    # If the matrix A has non-zero elements in the upper triangular part (including the
    # diagonal) then it is an implicit scheme. We do make a distinction between diagonally
    # implicit (nonzero diagonal but all other uppper triangular entries are zero) and
    # fully implicit schemes.
    is_implicit = false
    is_diagonally_implicit = false
    for i in 1:num_stages
        for j in i:num_stages
            if i == j && A[i, j] != zero(eltype(A))
                is_diagonally_implicit = true
            elseif A[i, j] != zero(eltype(A))
                is_implicit = true
            end
        end
    end

    U = ones(SMatrix{num_stages, 1})
    V = ones(SMatrix{1, 1})

    time_levels = TimeLevels(
        [0], # y
        [], # Δt G
        [],  # Δt F
    )

    if is_implicit
        return Implicit(A, SMatrix{1, num_stages}(B'), U, V, C, time_levels, order)
    elseif is_diagonally_implicit
        return DiagonallyImplicit(A, SMatrix{1, num_stages}(B'), U, V, C, time_levels, order)
    else
        return Explicit(A, SMatrix{1, num_stages}(B'), U, V, C, time_levels, order)
    end
end

"""
mapMultiStepToScheme::(Array{Float64}, Array{Float64}) -> AbstractTimeIntegrator \\
𝐲ₙ = ∑ from i=1 to r: αᵢ*𝐲ₙ₋ᵢ + Δt ∑ from i=0 to r: βᵢ*𝐅ₙ₋ᵢ \\
alpha::Array{Float64} is a vector that start at n=1 \\
alpha::Array{Float64} is a vector that start at n=0 \\

Implementation from article:
Vos, Peter & Eskilsson, Claes & Bolis, Alessandro & Chun, Sehun & Kirby, Robert & Sherwin, Spencer. (2011).
A generic framework for time-stepping partial differential equations (PDEs):
General linear methods, object-oriented implementation and application to fluid problems.
International Journal of Computational Fluid Dynamics. 25. 107-125. 10.1080/10618562.2011.575368.

Specficly Appendix A.2
"""
function mapMultiStepToScheme(
    α::SVector{Nα, Float64}, β::SVector{Nβ, Float64}, order::Int
) where {Nα, Nβ}
    N = Nα + Nβ - 1
    is_implicit::Bool = β[1] ≉ 0.0

    A = SMatrix{1,1}(β[1])

    B = SMatrix{N, 1}(
        if i == 1
            β[1]
        elseif i == Nα + 1
            1.0
        else
            0.0
        end for i in 1:N
    )

    U = SMatrix{1, N, Float64}(α..., β[2:end]...)

    V = SMatrix{N, N}(
        if i == 1
            U[j]
        elseif i == j + 1 && j > Nα
            1.0
        else
            0.0
        end for i in 1:N, j in 1:N
    )

    if is_implicit
        time_levels = TimeLevels(
            collect(0:(Nα - 1)), # y
            collect(0:(Nβ - 2)), # Δt G
            [], # Δt F
        )
        return Implicit(A, B, U, V, SVector(1.0), time_levels, order)
    end

    time_levels = TimeLevels(
        collect(0:(Nα - 1)), # y
        [], # Δt G
        collect(0:(Nβ - 2)),  # Δt F
    )
    return Explicit(A, B, U, V, SVector(0.0), time_levels, order)
end

"""
    mapIMEXMultiStageToScheme::(Matrix{Float64}, Matrix{Float64}, Array{Float64}, Array{Float64}, Array{Float64}, Int) -> IMEX \\

    A_IM::Matrix{Float64} is a square matrix of size sxs, where num_stages is the number of stages of the implicit scheme \\
    A_EX::Matrix{Float64} is a square matrix of size sxs, where num_stages is the number of stages of the explicit scheme \\
    B_IM::Array{Float64} is a vector of size num_stages \\
    B_EX::Array{Float64} is a vector of size num_stages \\
    C::Array{Float64} is a vector of size num_stages \\
    order::Int is the order of the scheme
"""
function mapIMEXMultiStageToScheme(
    A_IM::SMatrix{S, S, Float64},
    A_EX::SMatrix{S, S, Float64},
    B_IM::SVector{S, Float64},
    B_EX::SVector{S, Float64},
    C::SVector{S, Float64},
    order::Int,
) where {S}
    # then calls mapButcherTableauToScheme twice for both the implicit and explicit part and combines them
    # @assert size(A_IM) == size(A_EX) "The matrices A_IM and A_EX must have the same size"
    # @assert size(A_IM, 1) == size(A_IM, 2) "The matrix A must be square"
    # time levels are the same for both implicit and explicit parts

    scheme_IM = mapButcherTableauToScheme(A_IM, B_IM, C, order)
    scheme_EX = mapButcherTableauToScheme(A_EX, B_EX, C, order)
    return IMEX(
        scheme_IM.A,
        scheme_EX.A,
        scheme_IM.B,
        scheme_EX.B,
        scheme_IM.U,
        scheme_IM.V,
        scheme_IM.C,
        scheme_EX.C,
        scheme_IM.time_levels,
        order,
    )
end

############## Runge-Kutta ################

# Explicit:
# OrderScheme is defined as abstract type o} end

# order 1
const FORWARD_EULER = mapButcherTableauToScheme(
    SMatrix{1,1}(0.0), SVector(1.0), SVector(0.0), 1
)
# order 2
const EXPLICIT_MIDPOINT = mapButcherTableauToScheme(
    SMatrix{2,2}(0.0, 1/2, 0.0, 0.0), SVector(0.0, 1.0), SVector(0.0, 1 / 2), 2
)

# order 2
const HEUN2 = mapButcherTableauToScheme(
    SMatrix{2,2}(0.0, 1.0, 0.0, 0.0), SVector(1 / 2, 1 / 2), SVector(0.0, 1.0), 2
)
# order 2
const RALSTON2 = mapButcherTableauToScheme(
    SMatrix{2,2}(0.0, 2/3, 0.0, 0.0), SVector(1 / 4, 3 / 4), SVector(0.0, 2 / 3), 2
)

# order 3
const HEUN3 = mapButcherTableauToScheme(
    SMatrix{3,3}(0.0, 1/3, 0.0, 0.0, 0.0, 2/3, 0.0, 0.0, 0.0),
    SVector(1 / 4, 0.0, 3 / 4),
    SVector(0.0, 1 / 3, 2 / 3),
    3,
)

# order 3
const RK3 = mapButcherTableauToScheme(
    SMatrix{3,3}(0.0, 1/2, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
    SVector(1 / 6, 2 / 3, 1 / 6),
    SVector(0.0, 1 / 2, 1.0),
    3,
)

# order 3
const RALTSON3 = mapButcherTableauToScheme(
    SMatrix{3,3}(0.0, 1/2, 0.0, 0.0, 0.0, 3/4, 0.0, 0.0, 0.0),
    SVector(2 / 9, 1 / 3, 4 / 9),
    SVector(0.0, 1 / 2, 3 / 4),
    3,
)

# order 3
# Van der Houwen's/Wray's third-order method
const VDHW3 = mapButcherTableauToScheme(
    SMatrix{3,3}(0.0, 8/15, 1/4, 0.0, 0.0, 5/12, 0.0, 0.0, 0.0),
    SVector(1 / 4, 0, 3 / 4),
    SVector(0.0, 8 / 15, 2 / 3),
    3,
)

# order 3
# Third-order Strong Stability Preserving Runge-Kutta
const SSPRK3 = mapButcherTableauToScheme(
    SMatrix{3,3}(0.0, 1.0, 1/4, 0.0, 0.0, 1/4, 0.0, 0.0, 0.0),
    SVector(1 / 6, 1 / 6, 2 / 3),
    SVector(0.0, 1.0, 1 / 2),
    3,
)

# order 4
const RK4 = mapButcherTableauToScheme(
    SMatrix{4,4}(
        0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ),
    SVector(1 / 6, 1 / 3, 1 / 3, 1 / 6),
    SVector(0.0, 0.5, 0.5, 1.0),
    4,
)

# order 4
# 3/8-rule fourth-order method
const RK4_3_8 = mapButcherTableauToScheme(
    SMatrix{4,4}(
        0.0, 1/3, -1/3, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ),
    SVector(1 / 8, 3 / 8, 3 / 8, 1 / 8),
    SVector(0.0, 1 / 3, 2 / 3, 1.0),
    4,
)

# order 4
const RALTSON4 = mapButcherTableauToScheme(
    SMatrix{4,4}(
        0.0,
        2/5,
        (-2889 + 1428*sqrt(5))/1024,
        (-3365 + 2094*sqrt(5))/6040,
        0.0,
        0.0,
        (3785 - 1620*sqrt(5))/1024,
        (-975 - 3046*sqrt(5))/2552,
        0.0,
        0.0,
        0.0,
        (467040 + 203968*sqrt(5))/240845,
        0.0,
        0.0,
        0.0,
        0.0,
    ),
    SVector(
        (263 + 24*sqrt(5)) / 1812,
        (125 - 1000*sqrt(5)) / 3828,
        (3426304 + 1661952*sqrt(5)) / 5924787,
        (30 - 4*sqrt(5)) / 123,
    ),
    SVector(0.0, 2 / 5, (14 - 3*sqrt(5)) / 16, 1.0),
    4,
)

# Implicit:

# order 1
const BACKWARD_EULER = mapButcherTableauToScheme(
    SMatrix{1,1}(1.0), SVector(1.0), SVector(1.0), 1
)

const RADAU_IA_1 = mapButcherTableauToScheme(
    SMatrix{1,1}(1.0), SVector(1.0), SVector(0.0), 1
)

# order 2
const IMPLICIT_MIDPOINT = mapButcherTableauToScheme(
    SMatrix{1,1}(0.5), SVector(1.0), SVector(1 / 2), 2
)

# order 2
# R. Alexander. Diagonally implicit runge-kutta methods for stiff odes. SIAM J. Numer. Anal., 14(6):1006–1021, 1977
const _α_DIRK2 = 1 - sqrt(2)/2
const DIRK2 = mapButcherTableauToScheme(
    SMatrix{2,2}(_α_DIRK2, 1-_α_DIRK2, 0.0, _α_DIRK2), SVector(1-_α_DIRK2, _α_DIRK2), SVector(_α_DIRK2, 1), 2
)

# order 3
# Crouzeix's two-stage, 3rd order Diagonally Implicit Runge–Kutta method:
# R. Alexander. Diagonally implicit runge-kutta methods for stiff odes. SIAM J. Numer. Anal., 14(6):1006–1021, 1977
const DIRK3 = mapButcherTableauToScheme(
    SMatrix{2,2}(1/2 + sqrt(3)/6, -sqrt(3)/3, 0.0, 1/2+sqrt(3)/6),
    SVector(1 / 2, 1 / 2),
    SVector(1 / 2 + sqrt(3) / 6, 1 / 2 - sqrt(3) / 6),
    3,
)

# order 3
const RADAU_IA_3 = mapButcherTableauToScheme(
    SMatrix{2,2}(1/4, 1/4, -1/4, 5/12), SVector(1 / 4, 3 / 4), SVector(0, 2 / 3), 3
)

# order 4
# Crouzeix's three-stage, 4th order Diagonally Implicit Runge–Kutta method:
# R. Alexander. Diagonally implicit runge-kutta methods for stiff odes. SIAM J. Numer. Anal., 14(6):1006–1021, 1977
const _α_DIRK4 = 2 / sqrt(3) * cos(pi / 18)
const DIRK4 = mapButcherTableauToScheme(
    SMatrix{3,3}((1+_α_DIRK4)/2, -_α_DIRK4/2, 1+_α_DIRK4, 0.0, (1+_α_DIRK4)/2, -(1.0 + 2*_α_DIRK4), 0.0, 0.0, (1+_α_DIRK4)/2),
    SVector(1 / (6 * _α_DIRK4^2), 1.0 - 1 / (3 * _α_DIRK4^2), 1 / (6 * _α_DIRK4^2)),
    SVector((1 + _α_DIRK4) / 2, 1 / 2, (1 - _α_DIRK4) / 2),
    4,
)

# order 4
const GAUSS_LEGENDRE_4 = mapButcherTableauToScheme(
    SMatrix{2,2}(1/4, 1/4+sqrt(3)/6, 1/4-sqrt(3)/6, 1/4),
    SVector(1 / 2, 1 / 2),
    SVector(1 / 2 - sqrt(3) / 6, 1 / 2 + sqrt(3) / 6),
    4,
)

# order 6
const GAUSS_LEGENDRE_6 = mapButcherTableauToScheme(
    SMatrix{3,3}(
        5/36,
        5 / 36+sqrt(15) / 24,
        5 / 36+sqrt(15) / 30,
        2 / 9-sqrt(15) / 15,
        2/9,
        2 / 9+sqrt(15) / 15,
        5 / 36-sqrt(15) / 30,
        5 / 36-sqrt(15) / 24,
        5/36,
    ),
    SVector(5 / 18, 4 / 9, 5 / 18),
    SVector(1 / 2 - sqrt(15) / 10, 1 / 2, 1 / 2 + sqrt(15) / 10),
    6,
)

############## Multi-step  ################

# Explicit:

# Adams-Bashforth
# yₙ = yₙ₋₁ + Δt f(yₙ₋₁)
const AB1 = mapMultiStepToScheme(SVector(1.0), SVector(0.0, 1.0), 1)

# yₙ = yₙ₋₁ + Δt (3/2 f(yₙ₋₁) - 1/2 f(yₙ₋₂))
const AB2 = mapMultiStepToScheme(SVector(1.0), SVector(0.0, 3 / 2, -1 / 2), 2)

# yₙ = yₙ₋₁ + Δt (23/12 f(yₙ₋₁) - 4/3 f(yₙ₋₂) + 5/12 f(yₙ₋₃))
const AB3 = mapMultiStepToScheme(
    SVector(1.0), SVector(0.0, 23 / 12, -4 / 3, 5 / 12), 3
)

# yₙ = yₙ₋₁ + Δt (55/24 f(yₙ₋₁) - 59/24 f(yₙ₋₂) + 37/24 f(yₙ₋₃) - 9/24 f(yₙ₋₄))
const AB4 = mapMultiStepToScheme(
    SVector(1.0), SVector(0.0, 55 / 24, -59 / 24, 37 / 24, -9 / 24), 4
)

# yₙ = yₙ₋₁ + Δt (1901/720 f(yₙ₋₁) - 2774/720 f(yₙ₋₂) + 2616/720 f(yₙ₋₃) - 1274/720 f(yₙ₋₄) + 251/720 f(yₙ₋₅))
const AB5 = mapMultiStepToScheme(
    SVector(1.0),
    SVector(0.0, 1901 / 720, -2774 / 720, 2616 / 720, -1274 / 720, 251 / 720),
    5,
)

# Implicit:

# Adams-Moulton
# yₙ = yₙ₋₁ + Δt f(yₙ)
const AM0 = mapMultiStepToScheme(SVector(1.0), SVector(1.0), 1)

# yₙ = yₙ₋₁ + Δt (1/2 g(yₙ) + 1/2 g(yₙ₋₁))
const AM1 = mapMultiStepToScheme(SVector(1.0), SVector(1 / 2, 1 / 2), 2)

# yₙ = yₙ₋₁ + Δt (5/12 g(yₙ) + 8/12 g(yₙ₋₁) - 1/12 g(yₙ₋₂))
const AM2 = mapMultiStepToScheme(SVector(1.0), SVector(5 / 12, 8 / 12, -1 / 12), 3)

# yₙ = yₙ₋₁ + Δt (9/24 g(yₙ) + 19/24 g(yₙ₋₁) - 5/24 g(yₙ₋₂) + 1/24 g(yₙ₋₃))
const AM3 = mapMultiStepToScheme(
    SVector(1.0), SVector(9 / 24, 19 / 24, -5 / 24, 1 / 24), 4
)

# yₙ = yₙ₋₁ + Δt (251/720 g(yₙ) + 646/720 g(yₙ₋₁) - 264/720 g(yₙ₋₂) + 106/720 g(yₙ₋₃) - 19/720 g(yₙ₋₄))
const AM4 = mapMultiStepToScheme(
    SVector(1.0), SVector(251 / 720, 646 / 720, -264 / 720, 106 / 720, -19 / 720), 5
)

# backward differentiation formulas
# yₙ = yₙ₋₁ + Δt f(yₙ)
const BD1 = mapMultiStepToScheme(SVector(1.0), SVector(1.0), 1)

# yₙ = 4/3 yₙ₋₁ - 1/3 yₙ₋₂ + 2/3 Δt f(yₙ)
const BD2 = mapMultiStepToScheme(SVector(4 / 3, -1 / 3), SVector(2 / 3), 2)

# yₙ = 18/11 yₙ₋₁ - 9/11 yₙ₋₂ + 2/11 yₙ₋₃ + 6/11 Δt f(yₙ)
const BD3 = mapMultiStepToScheme(
    SVector(18 / 11, -9 / 11, 2 / 11), SVector(6 / 11), 3
)

# yₙ = 48/25 yₙ₋₁ - 36/25 yₙ₋₂ + 16/25 yₙ₋₃ - 3/25 yₙ₋₄ + 12/25 Δt f(yₙ)
const BD4 = mapMultiStepToScheme(
    SVector(48 / 25, -36 / 25, 16 / 25, -3 / 25), SVector(12 / 25), 4
)

# Almost Runge-Kutta methods
# https://ebookcentral-proquest-com.tudelft.idm.oclc.org/lib/delft/reader.action?docID=4591869&ppg=425
const ARK4 = Explicit(
    SMatrix{4,4}(0.0, 1/16, -1/4, 0.0, 0.0, 0.0, 2.0, 2/3, 0.0, 0.0, 0.0, 1/6, 0.0, 0.0, 0.0, 0.0),
    SMatrix{3,4}(0.0, 0.0, -1/3, 2/3, 0.0, 0.0, 1/6, 0.0, -2/3, 0.0, 1.0, 2.0),
    SMatrix{4,3}(1.0, 1.0, 1.0, 1.0, 1.0, 7/16, -3/4, 1/6, 1/2, 1/16, -1/4, 0.0),
    SMatrix{3,3}(1.0, 0.0, 0.0, 1/6, 0.0, -1.0, 0.0, 0.0, 0.0),
    SVector(0.0, 1 / 4, 1 / 2, 1.0), # TODO: not sure
    # inputs: yₙ₋₁ and Δt y'ₙ₋₁ and Δt² y''ₙ₋₁,
    TimeLevels(
        [0], # y
        [], # Δt G
        [0, 1],  # Δt F
    ),
    4,
)

##############    IMEX     ################

# Multi-stage
const BACKWARD_FORWARD_EULER = mapIMEXMultiStageToScheme(
    SMatrix{2,2}(0.0, 0.0, 0.0, 1.0),
    zeros(SMatrix{2,2}),
    SVector(0.0, 1.0),
    SVector(1.0, 0.0),
    SVector(0.0, 1.0),
    1,
)

# yₙ = yₙ₋₁ + Δt g(yₙ/2 + yₙ₋₁/2) + Δt f(yₙ₋₁ + Δt/2 f(yₙ₋₁))
const MIDPOINT_IMEX = mapIMEXMultiStageToScheme(
    SMatrix{2,2}(0.0, 0.0, 0.0, 1/2),
    SMatrix{2,2}(0.0, 1/2, 0.0, 0.0),
    SVector(0.0, 1.0),
    SVector(0.0, 1.0),
    SVector(0.0, 1 / 2),
    2,
)

const _γ = (3 + sqrt(3)) / 6
const RK3_IMEX = mapIMEXMultiStageToScheme(
    SMatrix{3,3}(0.0, 0.0, 0.0, 0.0, _γ, 1.0-2.0*_γ, 0.0, 0.0, _γ),
    SMatrix{3,3}(0.0, _γ, 1.0-2.0*_γ, 0.0, 0.0, 2.0*(1.0-_γ), 0.0, 0.0, 0.0),
    SVector(0.0, 1 / 2, 1 / 2),
    SVector(0.0, 1 / 2, 1 / 2),
    SVector(0.0, _γ, 1 - _γ),
    3,
)

# Multi-step

# Second-order Crank-Nicolson/Adams-Bashforth linear multistep scheme
# yₙ = yₙ₋₁ + Δt (1/2 g(yₙ) + 1/2 g(yₙ₋₁) + 3/2 f(yₙ₋₁) - 1/2 f(yₙ₋₂))
const CNAB2 = IMEX(
    SMatrix{1,1}(1/2),
    SMatrix{1,1}(0.0),
    SMatrix{4,1}(1 / 2, 1.0, 0.0, 0.0),
    SMatrix{4,1}(0.0, 0.0, 1.0, 0.0),
    SMatrix{1,4}(1.0, 1/2, 3/2, -1/2),
    SMatrix{4,4}(1.0, 0.0, 0.0, 0.0, 1/2, 0.0, 0.0, 0.0, 3/2, 0.0, 0.0, 1.0, -1/2, 0.0, 0.0, 0.0),
    SVector(0.0),  #TODO??
    SVector(0.0),   #TODO??
    TimeLevels(
        [0], # y
        [0], # Δt G
        [0, 1],  # Δt F
    ),
    2,
)

# Stiffly stable splitting scheme
const SSSS3 = IMEX(
    SMatrix{1,1}(2/3),
    SMatrix{1,1}(0.0),
    SMatrix{4,1}(2 / 3, 0.0, 0.0, 0.0),
    SMatrix{4,1}(0.0, 0.0, 1.0, 0.0),
    SMatrix{1,4}(4/3, -1/3, 4/3, -2/3),
    SMatrix{4,4}(4/3, 1.0, 0.0, 0.0, -1/3, 0.0, 0.0, 0.0, 4/3, 0.0, 0.0, 1.0, -2/3, 0.0, 0.0, 0.0),
    SVector(1.0),
    SVector(0.0),
    TimeLevels(
        [0, 1], # y
        [], # Δt G
        [0, 1],  # Δt F
    ),
    3,
)
