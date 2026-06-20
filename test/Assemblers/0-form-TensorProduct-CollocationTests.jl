module ZeroFormTensorProductCollocationTests

using Mantis

using Test
using LinearAlgebra

# Manufactured solution u = ∏ᵢ sin(ω xᵢ) and the matching collocation forcing.
#
# In Mantis the *pointwise* codifferential `δ(d(u⁰))` evaluates to the analyst Laplacian
# `+Δu = +∑ᵢ ∂²u/∂xᵢ²`. The Poisson problem `-Δu = f` is therefore collocated as
# `δ(d(u⁰)) = -f`. For u = ∏ᵢ sin(ω xᵢ) we have `-Δu = manifold_dim · ω² · u`, so the
# right-hand side field is `-f = -manifold_dim · ω² · u`.
function manufactured_forms(
    form_space::Forms.AbstractFormSpace{manifold_dim}
) where {manifold_dim}
    geometry = Forms.get_geometry(form_space)
    ω = 2.0 * pi
    function sol(x::Matrix{Float64})
        return [vec(prod(sin.(ω .* x); dims=2))]
    end
    function rhs(x::Matrix{Float64})
        y = prod(sin.(ω .* x); dims=2)
        return [vec(@. -manifold_dim * ω * ω * y)]
    end
    uₑ = Forms.AnalyticalFormField(0, sol, geometry, "u")
    f⁰ = Forms.AnalyticalFormField(0, rhs, geometry, "f")
    return uₑ, f⁰
end

# Solve `-Δu = f` by Greville collocation and return the discrete and exact solutions.
function solve_collocation_poisson(form_space, bc_value=0.0)
    uₑ, f⁰ = manufactured_forms(form_space)
    points = Assemblers.GrevilleCollocation(Forms.get_fe_space(form_space))
    inputs = Assemblers.CollocationInputs(form_space, f⁰, points)
    u⁰ = Assemblers.get_trial_form(inputs)
    collocation_form = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)
    bc = Forms.set_dirichlet_boundary_conditions(form_space, bc_value)
    A, b = Assemblers.assemble(collocation_form, bc)
    uₕ = Forms.build_form_field(form_space, vec(A \ b))
    return uₕ, uₑ, bc
end

# L2 error of the collocation solution on a `num_elements` mesh of degree-`p` C^{p-1} splines.
function collocation_l2_error(origin, box, num_elements, p)
    B = FunctionSpaces.create_bspline_space(origin, box, num_elements, p, p .- 1)
    Λ⁰ = Forms.FormSpace(0, B, "u")
    uₕ, uₑ, _ = solve_collocation_poisson(Λ⁰)
    geometry = Forms.get_geometry(Λ⁰)
    quad_degrees = (p isa Tuple ? p : (p,)) .+ 3
    canonical_qrule = Quadrature.tensor_product_rule(
        quad_degrees, Quadrature.gauss_legendre
    )
    dΩ = Quadrature.StandardQuadrature(
        canonical_qrule, Geometry.get_num_elements(geometry)
    )
    return Analysis.compute_error_total(uₕ, uₑ, dΩ, "L2")
end

@testset "GrevilleCollocation point construction" begin
    B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2, 2))
    points = Assemblers.GrevilleCollocation(B)
    num_basis = FunctionSpaces.get_num_basis(B)

    # Square system: one collocation point per basis function.
    @test Assemblers.get_num_points(points) == num_basis
    @test Assemblers.get_num_elements(points) ==
        Geometry.get_num_elements(FunctionSpaces.get_parametric_geometry(B))
    @test Assemblers.is_bijective(points)

    # Each basis function's point lies in exactly one element; together they cover 1:num_basis.
    all_ids = Int[]
    for e in 1:Assemblers.get_num_elements(points)
        ids = Assemblers.get_point_ids(points, e)
        append!(all_ids, ids)
        # Local coordinates are in the canonical [0, 1] cell.
        xi = Assemblers.get_collocation_points(points, e)
        for i in 1:Points.get_num_points(xi)
            @test all(0.0 .<= xi[i] .<= 1.0)
        end
    end
    @test sort(all_ids) == collect(1:num_basis)
end

@testset "1D Poisson convergence" begin
    p = 3
    errors = [collocation_l2_error(0.0, 1.0, n, p) for n in (8, 16, 32, 64)]
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test issorted(errors; rev=true)                 # error decreases under refinement
    @test errors[end] < 1e-3                          # accurate on the finest mesh
    @test rates[end] > p - 1.5                        # near the expected collocation rate
end

@testset "2D Poisson convergence" begin
    p = (3, 3)
    errors = [
        collocation_l2_error((0.0, 0.0), (1.0, 1.0), (n, n), p) for n in (8, 16, 32)
    ]
    rates = log2.(errors[1:(end - 1)] ./ errors[2:end])
    @test issorted(errors; rev=true)
    @test errors[end] < 1e-2
    @test rates[end] > 1.5
end

@testset "Dirichlet boundary conditions" begin
    B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2, 2))
    Λ⁰ = Forms.FormSpace(0, B, "u")
    bc_value = 0.5
    uₕ, _, bc = solve_collocation_poisson(Λ⁰, bc_value)
    # Each boundary basis coefficient is set exactly to the prescribed value, since the
    # corresponding collocation row is replaced by the identity equation.
    for idx in keys(bc)
        @test isapprox(uₕ.coefficients[idx], bc_value; atol=1e-12)
    end
end

@testset "UserCollocation and boundary-condition guard" begin
    B = FunctionSpaces.create_bspline_space(0.0, 1.0, 8, 3, 2)
    num_basis = FunctionSpaces.get_num_basis(B)
    pts = (collect(LinRange(0.0, 1.0, num_basis)),)
    user_points = Assemblers.UserCollocation(B, pts)

    @test !Assemblers.is_bijective(user_points)
    @test Assemblers.get_num_points(user_points) == num_basis

    # Boundary conditions require a point-to-basis bijection; they must be rejected here.
    Λ⁰ = Forms.FormSpace(0, B, "u")
    _, f⁰ = manufactured_forms(Λ⁰)
    inputs = Assemblers.CollocationInputs(Λ⁰, f⁰, user_points)
    u⁰ = Assemblers.get_trial_form(inputs)
    collocation_form = Assemblers.CollocationForm(((δ(d(u⁰)),),), ((f⁰,),), inputs)
    bc = Forms.set_dirichlet_boundary_conditions(Λ⁰, 0.0)
    @test_throws ArgumentError Assemblers.assemble(collocation_form, bc)
    # Without boundary conditions the same point set assembles fine.
    A, b = Assemblers.assemble(collocation_form)
    @test size(A) == (num_basis, num_basis)
end

end
