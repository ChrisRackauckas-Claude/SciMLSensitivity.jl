# Allocation regression tests for the continuous adjoints.

#
# Guards the two properties measured on master (Julia 1.10, Enzyme 0.13):
#   1. The per-call ceilings of the adjoint inner loops (vec_pjac! /
#      the Gauss integrand) with EnzymeVJP.
#   2. The per-step allocation slope of full adjoint solves: allocations must
#      not grow faster than the recorded slope with integration length —
#      catches any accidentally introduced per-step/per-stage allocation.
#   3. With `dgrad === nothing` (Gauss/Quadrature adjoint rhs), the parameter
#      shadow buffer must remain untouched — proves Enzyme is not computing a
#      discarded parameter gradient on every solver stage.
using SciMLSensitivity, OrdinaryDiffEq, LinearAlgebra, Test

# Allocation counts are meaningless under coverage instrumentation.
coverage_on = Base.JLOptions().code_coverage != 0

function lv!(du, u, p, t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
    return nothing
end
dgdu(out, u, p, t, i) = (out .= 1.0)

fwdsol(T) = solve(
    ODEProblem(lv!, [1.0, 1.0], (0.0, T), [1.5, 1.0, 3.0, 1.0]),
    Tsit5(); abstol = 1.0e-8, reltol = 1.0e-8, dense = true
)

function adjoint_allocs(T, sa)
    sol = fwdsol(T)
    ts = collect(range(0.0, T, length = 21))
    f() = adjoint_sensitivities(
        sol, Tsit5(); t = ts, dgdu_discrete = dgdu,
        sensealg = sa, abstol = 1.0e-6, reltol = 1.0e-3
    )
    f(); f()  # compile + warm caches
    return minimum(@allocated(f()) for _ in 1:5), length(sol.t)
end

@testset "adjoint allocation regressions" begin
    if coverage_on
        @info "coverage instrumentation active — skipping allocation regression tests"
    else
        @testset "per-call ceiling: GaussIntegrand vec_pjac! (EnzymeVJP)" begin
            sol = fwdsol(10.0)
            sa = GaussAdjoint(autojacvec = EnzymeVJP())
            S = SciMLSensitivity.GaussIntegrand(sol, sa, collect(sol.t), nothing)
            out = zeros(4); λ = [1.0, 2.0]; y = copy(sol.u[10]); t = sol.t[10]
            SciMLSensitivity.vec_pjac!(out, λ, y, t, S)
            SciMLSensitivity.vec_pjac!(out, λ, y, t, S)
            a = minimum(@allocated(SciMLSensitivity.vec_pjac!(out, λ, y, t, S)) for _ in 1:5)
            # measured 64 B (3 small allocs inside Enzyme.autodiff) on
            # Julia 1.10 / Enzyme 0.13; the ceiling leaves 2x headroom.
            @test a <= 128
        end

        @testset "per-step slope of full adjoint solves" begin
            # slope = (allocs(T=100) - allocs(T=10)) / (fwd-steps difference).
            # Measured on master: Gauss+Enzyme ~24 B/step, Interp+Enzyme 0 B/step.
            for (name, sa, ceiling) in [
                    ("Gauss+EnzymeVJP", GaussAdjoint(autojacvec = EnzymeVJP()), 64.0),
                    ("Interp+EnzymeVJP", InterpolatingAdjoint(autojacvec = EnzymeVJP()), 8.0),
                ]
                a10, n10 = adjoint_allocs(10.0, sa)
                a100, n100 = adjoint_allocs(100.0, sa)
                slope = (a100 - a10) / (n100 - n10)
                @test slope <= ceiling
            end
        end

        @testset "dgrad === nothing skips the parameter gradient (EnzymeVJP)" begin
            # The Gauss/Quadrature adjoint rhs calls vecjacobian! without dgrad on
            # every solver stage; Enzyme must receive `Const(p)` there. If the
            # parameter shadow were passed as `Duplicated`, it would be zeroed and
            # written each call — the NaN sentinel would not survive.
            sol = fwdsol(10.0)
            sa = GaussAdjoint(autojacvec = EnzymeVJP())
            adj_prob, _, _ = SciMLSensitivity.ODEAdjointProblem(
                sol, sa, Tsit5(),
                SciMLSensitivity.GaussIntegrand(sol, sa, collect(sol.t), nothing),
                nothing, collect(range(0.0, 10.0, length = 21)), dgdu
            )
            S = adj_prob.f.f  # ODEGaussAdjointSensitivityFunction
            tmp2 = S.diffcache.paramjac_config[2]
            dλ = zeros(2); λ = [1.0, 2.0]; y = copy(sol.u[10])
            SciMLSensitivity.vecjacobian!(dλ, y, λ, sol.prob.p, sol.t[10], S)
            fill!(tmp2, NaN)
            SciMLSensitivity.vecjacobian!(dλ, y, λ, sol.prob.p, sol.t[10], S)
            @test all(isnan, tmp2)
            @test all(isfinite, dλ)
        end
    end
end
