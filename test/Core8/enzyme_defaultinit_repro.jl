# Minimal reproducer for the #1469 follow-up Enzyme bug, reduced from `mtk.jl`:
# a single `DefaultInit` setup differentiated once by outer Enzyme reverse, through the
# Enzyme-native `_init_originator_gradient(::EnzymeOriginator)` init path. The init
# sensitivity differentiates ModelingToolkitBase's `CopyParamsByTemplate`, which on the
# affected (GitHub Actions x64) hardware miscompiles in Enzyme's augmented-forward with a
# wrong-length shadow:
#
#   BoundsError: attempt to access MemoryRef{Float64} at index [6]/[7]
#     __apply_copy_template / CopyParamsByTemplate / PromoteToTunableEltype
#     under runtime_generic_augfwd over
#       mapreduce(Base.Fix1(__apply_copy_template, ::NonlinearSolution), vcat, ...)
#
# This is the single-setup, single-call core of the `mtk.jl` "MTK Forward Mode" sweep
# (no 14-setup sweep, no Tracker, no Mooncake, short tspan). It only fires with the
# Enzyme-native init path; on the #1467 Zygote routing the init is never Enzyme-
# differentiated and this passes. Computes a finite gradient on AMD znver2 locally.

using ModelingToolkit, OrdinaryDiffEq
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqCore: DefaultInit
using SciMLSensitivity
using Enzyme
import SciMLStructures as SS
using SymbolicIndexingInterface: parameter_values
using Test

@parameters σ ρ β
@variables x(t) y(t) z(t) w(t) w2(t)

# The algebraic equation `0 ~ x^2 + y^2 - w2^2` forces DAE initialization.
eqs = [
    D(D(x)) ~ σ * (y - x),
    D(y) ~ x * (ρ - z) - y,
    D(z) ~ x * y - β * z,
    w ~ x + y + z + 2 * β,
    0 ~ x^2 + y^2 - w2^2,
]
@mtkbuild sys = ODESystem(eqs, t)

u0 = [D(x) => 2.0, x => 1.0, y => 0.0, z => 0.0]
p = [σ => 28.0, ρ => 10.0, β => 8 / 3]
tspan = (0.0, 1.0)

prob = ODEProblem(sys, u0, tspan, p; jac = true, guesses = [w2 => 0.0])
tunables, _, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob))

const _SENSEALG = GaussAdjoint(; autojacvec = SciMLSensitivity.ReverseDiffVJP())

function _repro_loss(tun, prob_)
    _, repack_, _ = SS.canonicalize(SS.Tunable(), parameter_values(prob_))
    new_prob = remake(prob_; u0 = prob_.u0, p = repack_(tun))
    sol = solve(
        new_prob, Rodas5P(); initializealg = DefaultInit(),
        sensealg = _SENSEALG, abstol = 1.0e-6, reltol = 1.0e-3,
    )
    return sum(sol)
end

@test begin
    dtunables = zero(tunables)
    diprob = Enzyme.make_zero(prob)
    Enzyme.autodiff(
        Enzyme.set_runtime_activity(Enzyme.Reverse),
        Enzyme.Const(_repro_loss), Enzyme.Active,
        Enzyme.Duplicated(tunables, dtunables),
        Enzyme.Duplicated(prob, diprob),
    )
    all(isfinite, dtunables) && any(!iszero, dtunables)
end
