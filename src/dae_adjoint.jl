# Adjoint sensitivity analysis for fully implicit DAEs (`DAEProblem`):
#
#     F(du, u, p, t) = 0
#
# following the augmented adjoint DAE formulation of Cao, Li, Petzold & Serban,
# "Adjoint sensitivity analysis for differential-algebraic equations: The adjoint
# DAE system and its numerical solution", SIAM Journal on Scientific Computing 24(3), 2003.
#
# For a cost functional G(p) = ∫ g(u, p, t) dt (discrete contributions are handled
# as Dirac deltas via callbacks), the adjoint λ(t) satisfies, in forward time,
#
#     d/dt(F_du' λ) - F_u' λ + g_u' = 0
#
# with dG/du₀ = (F_du' λ)(t₀) and dG/dp = ∫ (g_p' - F_p' λ) dt. To avoid the total
# time derivative of F_du along the trajectory, introduce the augmented variable
#
#     w := F_du' λ
#
# which turns the adjoint into the semi-explicit system
#
#     ẇ = F_u' λ - g_u',        0 = w - F_du' λ.
#
# As written, this system is index-2 even when the original DAE is index-1 (λ
# components paired with algebraic variables appear only under one differentiation
# of the constraint). We therefore index-reduce structurally, using the fact that
# w_j ≡ 0 for every algebraic variable j (zero column of F_du): for those rows the
# differential equation ẇ_j = 0 = [F_u' λ - g_u']_j is itself the algebraic
# equation determining the remaining adjoint components. With the parameter
# quadrature appended, the augmented adjoint state is z = [w; λ; grad] with
#
#     row j ∈ differential vars:  ẇ_j    = [F_u' λ]_j - g_u[j]
#     row j ∈ algebraic vars:     0      = w_j
#     row n+j, j differential:    0      = w_j - [F_du' λ]_j
#     row n+j, j algebraic:       0      = [F_u' λ]_j - g_u[j]
#     grad rows:                  grad'  = F_p' λ - g_p'
#
# solved backwards from z(T) = 0 as a singular-mass-matrix ODEProblem. For the
# special case F = M u̇ - f(u, p, t) this reduces exactly to the transposed
# mass-matrix adjoint used by `ODEAdjointProblem`. Note no sign flips relative to
# the ODE adjoint code appear here: the VJPs of the residual F carry them.
#
# Discrete cost contributions jump `w` at the data points. When the cost touches
# algebraic variables (index-1) or when the DAE is index-2 Hessenberg, the raw jump
# gᵤ is inconsistent with the adjoint's (hidden) constraints and must be corrected.
# Both cases reduce to the same structure used by the semi-explicit mass-matrix
# code in `ReverseLossCallback`: a multiplier Δλa supported on the algebraic
# equations produces the consistent jump
#
#     w[diffvars] += gᵤ[diffvars] + dhdd' Δλa,      dhdd = F_u[algeqs, diffvars]
#
# together with the parameter-gradient correction dp += Δλa' ∂F_algeqs/∂p that is
# accumulated in `_adjoint_sensitivities` (mirroring `rcb.Δλas`). The multiplier is
# index-dependent:
#
#     index-1:            Δλa = -dhda' \ gᵤ[algvars],   dhda = F_u[algeqs, algvars]
#     index-2 Hessenberg: Δλa = -(N dhdd') \ (N gᵤ[diffvars]),
#                         N = (F_du[diffeqs, diffvars] \ F_u[diffeqs, algvars])'
#
# where the index-2 formula enforces the adjoint hidden constraint (post-jump
# w[diffvars] must annihilate range(F_u[diffeqs, algvars]) through F_du) while
# leaving the pairing with admissible state perturbations (tangent to the
# constraint manifold) unchanged. Discrete cost on the algebraic variables of an
# index-2 DAE would require derivatives of delta distributions and is rejected.

# Structural index information for the implicit DAE, detected once from the
# Jacobians at the terminal time and `differential_vars`.
struct DAEAdjointStructure
    diffvar_idxs::Vector{Int}
    algevar_idxs::Vector{Int}
    diffeq_idxs::Vector{Int}
    algeeq_idxs::Vector{Int}
    index2::Bool
end

struct DAEAdjointDiffCache{S <: DAEAdjointStructure, PJC, PG, PGC, DGV, DG1, DG2, T, R}
    structure::S
    paramjac_config::PJC
    pg::PG
    pg_config::PGC
    dg_val::DGV
    dgdu::DG1
    dgdp::DG2
    tunables::T
    repack::R
end

function dae_du_jacobian(f, isinplace, dy, y, p, t)
    return if isinplace
        ForwardDiff.jacobian(dy) do du
            res = similar(du, promote_type(eltype(du), eltype(y)), length(y))
            res .= false
            f(res, du, y, p, t)
            res
        end
    else
        ForwardDiff.jacobian(du -> vec(f(du, y, p, t)), dy)
    end
end

function dae_u_jacobian(f, isinplace, dy, y, p, t)
    return if isinplace
        ForwardDiff.jacobian(y) do u
            res = similar(u, promote_type(eltype(u), eltype(dy)), length(u))
            res .= false
            f(res, dy, u, p, t)
            res
        end
    else
        ForwardDiff.jacobian(u -> vec(f(dy, u, p, t)), y)
    end
end

function dae_p_jacobian(f, isinplace, dy, y, p, t, tunables, repack)
    return if isinplace
        ForwardDiff.jacobian(tunables) do _tunables
            res = similar(_tunables, length(y))
            res .= false
            f(res, dy, y, repack(_tunables), t)
            res
        end
    else
        ForwardDiff.jacobian(_tunables -> vec(f(dy, y, repack(_tunables), t)), tunables)
    end
end

const DAE_ADJOINT_UNBALANCED_MESSAGE = """
The number of algebraic equations (rows of ∂F/∂(du) that are identically zero)
does not match the number of algebraic variables (from `differential_vars`, or
columns of ∂F/∂(du) that are identically zero). Adjoint sensitivity analysis for
`DAEProblem` requires a square algebraic subsystem. If `differential_vars` was not
provided to the `DAEProblem`, provide it explicitly so that the algebraic
variables can be identified structurally.
"""

const DAE_ADJOINT_HIGH_INDEX_MESSAGE = """
The algebraic subsystem of the DAE could not be classified as index-1 (∂F_alg/∂u_alg
nonsingular) or Hessenberg index-2 (∂F_alg/∂u_alg ≡ 0 with
∂F_alg/∂u_diff * (∂F_diff/∂(du)_diff)⁻¹ * ∂F_diff/∂u_alg nonsingular) at the
terminal time of the forward solution. Mixed index-1/index-2 systems and DAEs of
index 3 or higher are not currently supported by the `DAEProblem` adjoint.
Consider index-reducing the system (e.g. via ModelingToolkit's `structural_simplify`)
before differentiating it.
"""

# Classify the DAE at (dy, y, p, t). The zero rows/columns of ∂F/∂(du) determine the
# algebraic equations/variables; `differential_vars` from the DAEProblem takes
# precedence for the variables when provided.
function dae_adjoint_structure(f, isinplace, dy, y, p, t, differential_vars)
    n = length(y)
    J_du = dae_du_jacobian(f, isinplace, dy, y, p, t)
    algeeq_idxs = findall(i -> all(iszero, @view(J_du[i, :])), 1:n)
    algevar_idxs = if differential_vars === nothing
        findall(j -> all(iszero, @view(J_du[:, j])), 1:n)
    else
        findall(!, collect(differential_vars))
    end
    diffeq_idxs = setdiff(1:n, algeeq_idxs)
    diffvar_idxs = setdiff(1:n, algevar_idxs)
    length(algeeq_idxs) == length(algevar_idxs) ||
        error(DAE_ADJOINT_UNBALANCED_MESSAGE)

    index2 = false
    if !isempty(algevar_idxs)
        J_u = dae_u_jacobian(f, isinplace, dy, y, p, t)
        dhda = J_u[algeeq_idxs, algevar_idxs]
        if !issuccess(lu(dhda, check = false))
            all(iszero, dhda) || error(DAE_ADJOINT_HIGH_INDEX_MESSAGE)
            Fdd = lu(J_du[diffeq_idxs, diffvar_idxs], check = false)
            issuccess(Fdd) || error(DAE_ADJOINT_HIGH_INDEX_MESSAGE)
            N = (Fdd \ J_u[diffeq_idxs, algevar_idxs])'
            NB = N * J_u[algeeq_idxs, diffvar_idxs]'
            issuccess(lu(NB, check = false)) || error(DAE_ADJOINT_HIGH_INDEX_MESSAGE)
            index2 = true
        end
    end
    return DAEAdjointStructure(
        diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs, index2
    )
end

struct DAEInterpolatingAdjointSensitivityFunction{
        C <: DAEAdjointDiffCache, Alg <: InterpolatingAdjoint,
        uType, duType, SType, pType, fType,
    } <: SensitivityFunction
    diffcache::C
    sensealg::Alg
    discrete::Bool
    y::uType
    dy::duType
    sol::SType
    prob::pType
    f::fType
end

function DAEInterpolatingAdjointSensitivityFunction(
        g, sensealg, discrete, sol, dgdu, dgdp, f, alg;
        tspan = reverse(sol.prob.tspan)
    )
    prob = sol.prob
    u0 = state_values(prob)
    p = parameter_values(prob)

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    elseif isscimlstructure(p)
        tunables, repack, _ = canonicalize(Tunable(), p)
    elseif isfunctor(p)
        error(
            "Functors.jl parameter structs are not supported for DAEProblem adjoints. " *
                "Make `p` an `AbstractArray` or a SciMLStructure."
        )
    else
        throw(SciMLStructuresCompatibilityError())
    end

    numstates = length(u0)
    numparams = p === nothing || p === SciMLBase.NullParameters() ? 0 : length(tunables)
    isinplace = DiffEqBase.isinplace(prob)
    unwrappedf = unwrapped_f(f)
    autojacvec = sensealg.autojacvec
    @assert autojacvec !== nothing

    _t = tspan[1] # forward-time terminal point
    y = copy(state_values(sol)[end])
    dy = similar(y)
    sol(dy, _t, Val{1})

    structure = dae_adjoint_structure(
        unwrappedf, isinplace, dy, y, p, _t, prob.differential_vars
    )

    if p === nothing || p isa SciMLBase.NullParameters
        _p = similar(y, (0,))
        _p .= false
    else
        _p = tunables
    end

    if autojacvec isa ReverseDiffVJP
        tape = if isinplace
            ReverseDiff.GradientTape((dy, y, _p, [_t])) do du, u, p, t
                res = p !== nothing && p !== SciMLBase.NullParameters() ?
                    similar(p, size(u)) : similar(u)
                res .= false
                unwrappedf(res, du, u, repack(p), first(t))
                return vec(res)
            end
        else
            ReverseDiff.GradientTape((dy, y, _p, [_t])) do du, u, p, t
                vec(unwrappedf(du, u, repack(p), first(t)))
            end
        end
        paramjac_config = compile_tape(autojacvec) ? ReverseDiff.compile(tape) : tape
    elseif autojacvec isa ZygoteVJP || autojacvec isa Bool
        autojacvec === true &&
            error("`autojacvec = true` is not supported for DAEProblem adjoints. Use `ReverseDiffVJP()`, `ZygoteVJP()`, or `autojacvec = false`.")
        paramjac_config = nothing
    else
        error(
            "$(nameof(typeof(autojacvec))) is not currently supported for DAEProblem adjoints. " *
                "Use `ReverseDiffVJP()`, `ZygoteVJP()`, or `autojacvec = false`."
        )
    end

    if !discrete
        if dgdu !== nothing
            pg = nothing
            pg_config = nothing
            if dgdp !== nothing
                dg_val = (similar(u0, numstates), similar(u0, numparams))
                dg_val[1] .= false
                dg_val[2] .= false
            else
                dg_val = similar(u0, numstates)
                dg_val .= false
            end
        else
            pgpu = UGradientWrapper(g, _t, p)
            pgpu_config = build_grad_config(sensealg, pgpu, u0, tunables)
            pgpp = ParamGradientWrapper(g, _t, u0)
            pgpp_config = build_grad_config(sensealg, pgpp, tunables, tunables)
            pg = (pgpu, pgpp)
            pg_config = (pgpu_config, pgpp_config)
            dg_val = (similar(u0, numstates), similar(u0, numparams))
            dg_val[1] .= false
            dg_val[2] .= false
        end
    else
        dg_val = nothing
        pg = nothing
        pg_config = nothing
    end

    diffcache = DAEAdjointDiffCache(
        structure, paramjac_config, pg, pg_config, dg_val,
        dgdu, dgdp, tunables, repack
    )

    return DAEInterpolatingAdjointSensitivityFunction(
        diffcache, sensealg, discrete, y, dy, sol, prob, f
    )
end

# z = [w; λ; grad], solved backwards. `dz` is used as scratch: the u-VJP is written
# into the w rows and the du-VJP into the λ rows, then both are rearranged in place
# into the residual layout derived at the top of this file.
function (S::DAEInterpolatingAdjointSensitivityFunction)(dz, z, p, t)
    (; y, dy, sol, discrete, diffcache) = S
    (; structure) = diffcache
    (; diffvar_idxs, algevar_idxs) = structure

    n = length(y)
    if t isa ForwardDiff.Dual && eltype(y) <: AbstractFloat
        _y = sol(t, continuity = :right)
        _dy = sol(t, Val{1}, continuity = :right)
    else
        sol(y, t, continuity = :right)
        sol(dy, t, Val{1}, continuity = :right)
        _y = y
        _dy = dy
    end

    w = @view z[1:n]
    λ = @view z[(n + 1):(2n)]
    vjp_u = @view dz[1:n]
    vjp_du = @view dz[(n + 1):(2n)]
    dgrad = @view dz[(2n + 1):length(dz)]

    dae_vecjacobian!(vjp_du, vjp_u, dgrad, _dy, _y, λ, p, t, S)

    # Continuous cost enters both the ẇ rows and the algebraic λ rows through the
    # same g_u subtraction, so apply it before the rows are rearranged.
    discrete || accumulate_dae_cost!(vjp_u, dgrad, _y, p, t, S)

    @inbounds for j in algevar_idxs
        dz[n + j] = dz[j] # 0 = [F_u' λ]_j - g_u[j]
        dz[j] = w[j]      # 0 = w_j
    end
    @inbounds for j in diffvar_idxs
        dz[n + j] = w[j] - dz[n + j] # 0 = w_j - [F_du' λ]_j
    end
    return nothing
end

function accumulate_dae_cost!(dw, dgrad, y, p, t, S)
    (; dgdu, dgdp, dg_val, pg, pg_config) = S.diffcache
    if dgdu !== nothing
        if dgdp === nothing
            dgdu(dg_val, y, p, t)
            dw .-= vec(dg_val)
        else
            dgdu(dg_val[1], y, p, t)
            dw .-= vec(dg_val[1])
            if !isempty(dgrad)
                dgdp(dg_val[2], y, p, t)
                dgrad .-= vec(dg_val[2])
            end
        end
    else
        pg[1].t = t
        pg[1].p = p
        gradient!(dg_val[1], pg[1], y, S.sensealg, pg_config[1])
        dw .-= vec(dg_val[1])
        if !isempty(dgrad)
            pg[2].t = t
            pg[2].u = y
            gradient!(dg_val[2], pg[2], p, S.sensealg, pg_config[2])
            dgrad .-= vec(dg_val[2])
        end
    end
    return nothing
end

function dae_vecjacobian!(vjp_du, vjp_u, vjp_p, dy, y, λ, p, t, S)
    return dae_vecjacobian!(
        vjp_du, vjp_u, vjp_p, dy, y, λ, p, t, S, S.sensealg.autojacvec
    )
end

function dae_vecjacobian!(
        vjp_du, vjp_u, vjp_p, dy, y, λ, p, t, S,
        autojacvec::ReverseDiffVJP
    )
    prob = getprob(S)
    f = unwrapped_f(S.f)
    (; tunables, repack) = S.diffcache
    u0 = state_values(prob)

    if p === nothing || p isa SciMLBase.NullParameters
        _p = similar(y, (0,))
        _p .= false
        _tunables = _p
    else
        _p = tunables
        _tunables = tunables
    end

    if eltype(λ) <: eltype(u0) && t isa eltype(u0) && compile_tape(autojacvec)
        tape = S.diffcache.paramjac_config
    else
        # Dual numbers from the (stiff) adjoint solver's AD require retaping with
        # promoted element types, mirroring `_vecjacobian!` for ODEs.
        _y = eltype(y) === eltype(λ) ? y : convert.(promote_type(eltype(y), eltype(λ)), y)
        _dy = eltype(dy) === eltype(λ) ? dy :
            convert.(promote_type(eltype(dy), eltype(λ)), dy)
        tape = if inplace_sensitivity(S)
            ReverseDiff.GradientTape((_dy, _y, _p, [t])) do du, u, p, t
                res = similar(u, size(u))
                res .= false
                f(res, du, u, repack(p), first(t))
                return vec(res)
            end
        else
            ReverseDiff.GradientTape((_dy, _y, _p, [t])) do du, u, p, t
                vec(f(du, u, repack(p), first(t)))
            end
        end
    end

    tdu, tu, tp, tt = ReverseDiff.input_hook(tape)
    output = ReverseDiff.output_hook(tape)
    ReverseDiff.unseed!(tdu)
    ReverseDiff.unseed!(tu)
    ReverseDiff.unseed!(tp)
    ReverseDiff.unseed!(tt)
    ReverseDiff.value!(tdu, dy)
    ReverseDiff.value!(tu, y)
    p isa SciMLBase.NullParameters || ReverseDiff.value!(tp, _tunables)
    ReverseDiff.value!(tt, [t])
    ReverseDiff.forward_pass!(tape)
    ReverseDiff.increment_deriv!(output, λ)
    ReverseDiff.reverse_pass!(tape)
    copyto!(vec(vjp_du), ReverseDiff.deriv(tdu))
    copyto!(vec(vjp_u), ReverseDiff.deriv(tu))
    isempty(vjp_p) || copyto!(vec(vjp_p), ReverseDiff.deriv(tp))
    return nothing
end

function dae_vecjacobian!(
        vjp_du, vjp_u, vjp_p, dy, y, λ, p, t, S,
        autojacvec::ZygoteVJP
    )
    inplace_sensitivity(S) &&
        error("`ZygoteVJP` requires an out-of-place DAE residual `f(du, u, p, t)`. Use `ReverseDiffVJP()` for in-place residuals.")
    f = unwrapped_f(S.f)
    (; tunables, repack) = S.diffcache

    if p === nothing || p isa SciMLBase.NullParameters
        _, back = Zygote.pullback(dy, y) do du, u
            vec(f(du, u, p, t))
        end
        tmp_du, tmp_u = back(λ)
        tmp_p = nothing
    else
        _, back = Zygote.pullback(dy, y, tunables) do du, u, _tunables
            vec(f(du, u, repack(_tunables), t))
        end
        tmp_du, tmp_u, tmp_p = back(λ)
    end

    if tmp_du !== nothing
        copyto!(vec(vjp_du), vec(tmp_du))
    elseif autojacvec.allow_nothing
        vjp_du .= false
    else
        throw(ZygoteVJPNothingError())
    end
    if tmp_u !== nothing
        copyto!(vec(vjp_u), vec(tmp_u))
    elseif autojacvec.allow_nothing
        vjp_u .= false
    else
        throw(ZygoteVJPNothingError())
    end
    if !isempty(vjp_p)
        if tmp_p !== nothing
            copyto!(vec(vjp_p), vec(tmp_p))
        elseif autojacvec.allow_nothing
            vjp_p .= false
        else
            throw(ZygoteVJPNothingError())
        end
    end
    return nothing
end

function dae_vecjacobian!(
        vjp_du, vjp_u, vjp_p, dy, y, λ, p, t, S,
        isautojacvec::Bool
    )
    f = unwrapped_f(S.f)
    isinplace = inplace_sensitivity(S)
    (; tunables, repack) = S.diffcache
    J_du = dae_du_jacobian(f, isinplace, dy, y, p, t)
    J_u = dae_u_jacobian(f, isinplace, dy, y, p, t)
    mul!(vec(vjp_du), J_du', λ)
    mul!(vec(vjp_u), J_u', λ)
    if !isempty(vjp_p)
        pJ = dae_p_jacobian(f, isinplace, dy, y, p, t, tunables, repack)
        mul!(vec(vjp_p), pJ', λ)
    end
    return nothing
end

# Fully implicit residual form of the augmented adjoint, M ż - G(z), so the
# adjoint can be solved as a `DAEProblem` by the same DAE algorithm as the forward
# pass (e.g. `DFBDF`). `mmdiag` is the diagonal of the augmented mass matrix.
struct DAEAdjointResidual{S <: DAEInterpolatingAdjointSensitivityFunction, M} <:
    SensitivityFunction
    S::S
    mmdiag::M
end

function (R::DAEAdjointResidual)(res, dz, z, p, t)
    R.S(res, z, p, t)
    @. res = R.mmdiag * dz - res
    return nothing
end

# Discrete-cost jump handling for the DAE adjoint, mirroring `ReverseLossCallback`
# with the index-1/index-2 corrections derived at the top of this file.
struct DAEReverseLossCallback{
        λType, timeType, yType, RefType, AlgType, DG1, DG2,
        cacheType, fType, solType, ΔλasType,
    }
    λ::λType
    t::timeType
    y::yType
    dy::yType
    cur_time::RefType
    idx::Int
    sensealg::AlgType
    dgdu::DG1
    dgdp::DG2
    diffcache::cacheType
    f::fType
    sol::solType
    Δλas::ΔλasType
    no_start::Bool
end

function DAEReverseLossCallback(sensefun, λ, t, dgdu, dgdp, cur_time, no_start)
    (; sensealg, y) = sensefun
    idx = length(state_values(getprob(sensefun)))
    Δλas = Tuple{Vector{eltype(λ)}, eltype(t)}[]
    return DAEReverseLossCallback(
        λ, t, y, similar(y), cur_time, idx, sensealg, dgdu, dgdp,
        sensefun.diffcache, sensefun.f, sensefun.sol, Δλas, no_start
    )
end

function (f::DAEReverseLossCallback)(integrator)
    (; λ, t, y, dy, cur_time, idx, dgdu, dgdp, sol, no_start) = f
    (; structure) = f.diffcache
    (; diffvar_idxs, algevar_idxs, diffeq_idxs, algeeq_idxs, index2) = structure

    no_start && cur_time[] == 1 && return nothing

    p, z = integrator.p, integrator.u
    n = idx
    ti = t[cur_time[]]
    sol(y, ti)

    gᵤ = @view λ[1:n]
    dgdu(gᵤ, y, p, ti, cur_time[])

    if dgdp !== nothing && length(λ) > n
        gp = @view λ[(n + 1):end]
        dgdp(gp, y, p, ti, cur_time[])
        z[(2n + 1):(2n + length(gp))] .+= gp
    end

    if isempty(algevar_idxs)
        z[1:n] .+= gᵤ
    else
        sol(dy, ti, Val{1})
        residual = unwrapped_f(f.f)
        isinplace = DiffEqBase.isinplace(sol.prob)
        J_u = dae_u_jacobian(residual, isinplace, dy, y, p, ti)
        dhdd = J_u[algeeq_idxs, diffvar_idxs]
        if index2
            all(iszero, @view(gᵤ[algevar_idxs])) ||
                error("Discrete cost contributions on the algebraic variables of an index-2 DAE are not supported: they correspond to derivatives of delta distributions in the adjoint. Reformulate the cost in terms of the differential variables.")
            J_du = dae_du_jacobian(residual, isinplace, dy, y, p, ti)
            N = (lu(J_du[diffeq_idxs, diffvar_idxs]) \ J_u[diffeq_idxs, algevar_idxs])'
            Δλa = -((N * dhdd') \ (N * gᵤ[diffvar_idxs]))
        else
            dhda = J_u[algeeq_idxs, algevar_idxs]
            Δλa = -(dhda' \ gᵤ[algevar_idxs])
        end
        z[diffvar_idxs] .+= @view(gᵤ[diffvar_idxs])
        z[diffvar_idxs] .+= dhdd' * Δλa
        push!(f.Δλas, (Δλa, ti))
    end

    derivative_discontinuity!(integrator, true)
    cur_time[] -= 1
    return nothing
end

function generate_dae_callbacks(
        sensefun, dgdu, dgdp, λ, t, t0, init_cb, terminated,
        no_start
    )
    init_cb || return CallbackSet(), nothing, nothing
    cur_time = Ref(length(t))

    _t, duplicate_iterator_times = separate_nonunique(t)
    rlcb = DAEReverseLossCallback(sensefun, λ, t, dgdu, dgdp, cur_time, no_start)
    if eltype(_t) !== typeof(t0)
        _t = convert.(typeof(t0), _t)
    end
    cb = PresetTimeCallback(_t, rlcb)

    if duplicate_iterator_times !== nothing
        cbrev_dupl_affect = DAEReverseLossCallback(
            sensefun, λ, t, dgdu, dgdp, cur_time, no_start
        )
        cb_dupl = PresetTimeCallback(duplicate_iterator_times[1], cbrev_dupl_affect)
        return CallbackSet(cb, cb_dupl), rlcb, duplicate_iterator_times
    else
        return CallbackSet(cb), rlcb, duplicate_iterator_times
    end
end

@doc doc"""
```julia
DAEAdjointProblem(sol, sensealg::InterpolatingAdjoint, alg, t = nothing,
                  dgdu_discrete = nothing, dgdp_discrete = nothing,
                  dgdu_continuous = nothing, dgdp_continuous = nothing,
                  g = nothing; kwargs...)
```

Constructs the adjoint problem for a fully implicit `DAEProblem` solution `sol`
(`F(du, u, p, t) = 0`), using the augmented adjoint DAE formulation of Cao, Li,
Petzold & Serban (SIAM Journal on Scientific Computing 24(3), 2003). The augmented adjoint state is
`z = [w; λ; grad]` with `w = (∂F/∂(du))' λ`, and the returned problem is a
singular-mass-matrix `ODEProblem` to be solved backwards in time with a
mass-matrix-capable stiff solver (e.g. `FBDF`, `Rodas5P`).

The arguments mirror `ODEAdjointProblem`. Index-1 DAEs are fully supported,
including discrete and continuous cost contributions on the algebraic variables.
Hessenberg index-2 DAEs are supported with cost contributions on the differential
variables (the consistent boundary jumps are obtained by projecting onto the
adjoint constraint manifold). Higher-index and mixed index-1/index-2 systems are
rejected; index-reduce such systems first. The forward solution must be dense
(`InterpolatingAdjoint` without checkpointing) and its interpolant must support
first-derivative evaluation `sol(t, Val{1})`, which holds for the native Julia DAE
solvers such as `DFBDF`.
"""
@noinline function DAEAdjointProblem(
        sol, sensealg::InterpolatingAdjoint, alg,
        t = nothing,
        dgdu_discrete::DG1 = nothing,
        dgdp_discrete::DG2 = nothing,
        dgdu_continuous::DG3 = nothing,
        dgdp_continuous::DG4 = nothing,
        g::G = nothing,
        ::Val{RetCB} = Val(false);
        checkpoints = current_time(sol),
        callback = CallbackSet(), no_start = false,
        reltol = nothing, abstol = nothing,
        kwargs...
    ) where {DG1, DG2, DG3, DG4, G, RetCB}
    dgdu_discrete === nothing && dgdu_continuous === nothing && g === nothing &&
        error("Either `dgdu_discrete`, `dgdu_continuous`, or `g` must be specified.")
    t !== nothing && dgdu_discrete === nothing && dgdp_discrete === nothing &&
        error("It looks like you're using the direct `adjoint_sensitivities` interface
               with a discrete cost function but no specified `dgdu_discrete` or `dgdp_discrete`.
               Please use the higher level `solve` interface or specify these two contributions.")
    ischeckpointing(sensealg, sol) &&
        error("Checkpointing is not currently supported for DAEProblem adjoints. Use a dense forward solution with `InterpolatingAdjoint(checkpointing = false)`.")
    if callback !== nothing &&
            !(
            callback isa CallbackSet && isempty(callback.continuous_callbacks) &&
                isempty(callback.discrete_callbacks)
        )
        error("Callbacks are not currently supported for DAEProblem adjoints.")
    end

    (; tspan) = sol.prob
    p = parameter_values(sol.prob)
    u0 = state_values(sol.prob)

    if p === nothing || p isa SciMLBase.NullParameters
        tunables, repack = p, identity
    else
        tunables, repack, _ = canonicalize(Tunable(), p)
    end

    terminated = false
    if hasfield(typeof(sol), :retcode)
        if sol.retcode == ReturnCode.Terminated
            tspan = (tspan[1], last(current_time(sol)))
            terminated = true
        end
    end
    tspan = reverse(tspan)

    discrete = (
        t !== nothing &&
            (
            dgdu_continuous === nothing && dgdp_continuous === nothing ||
                g !== nothing
        )
    )

    numstates = length(u0)
    numparams = p === nothing || p === SciMLBase.NullParameters() ? 0 : length(tunables)

    λ = p === nothing || p === SciMLBase.NullParameters() ? similar(u0) :
        one(eltype(u0)) .* similar(tunables, numstates + numparams)
    λ .= false

    sense = DAEInterpolatingAdjointSensitivityFunction(
        g, sensealg, discrete, sol,
        dgdu_continuous, dgdp_continuous,
        sol.prob.f, alg; tspan
    )

    init_cb = (discrete || dgdu_discrete !== nothing)
    cb, rcb, _ = generate_dae_callbacks(
        sense, dgdu_discrete, dgdp_discrete,
        λ, t, tspan[2], init_cb, terminated, no_start
    )

    len = 2 * numstates + numparams
    z0 = similar(λ, len)
    z0 .= false

    mmdiag = zeros(eltype(z0), len)
    mmdiag[sense.diffcache.structure.diffvar_idxs] .= one(eltype(z0))
    mmdiag[(2 * numstates + 1):len] .= one(eltype(z0))

    # The algebraic λ rows must be (re-)solved consistently at the terminal time and
    # after every discrete-cost jump; `BrownFullBasicInit` does exactly that and its
    # initialization system is the adjoint consistency condition of Cao et al. For
    # Hessenberg index-2 DAEs that system is singular (the index-2 adjoint multiplier
    # only appears under differentiation), so skip initialization and let the BDF
    # discretization determine the multiplier from the second step on; the terminal
    # λ value is never used by the gradient extraction.
    initializealg = sense.diffcache.structure.index2 ? SciMLBase.NoInit() :
        BrownFullBasicInit()

    adj_prob = if alg isa SciMLBase.AbstractDAEAlgorithm
        # Solve the augmented adjoint in fully implicit residual form with the
        # user's DAE algorithm.
        resid = DAEAdjointResidual(sense, mmdiag)
        diffvars_aug = mmdiag .== one(eltype(z0))
        dz0 = zero(z0)
        daefun = DAEFunction{true, true}(resid)
        # Automatic initial-dt selection does not support backwards-in-time
        # DAEProblems, so seed the reverse solve with the forward solution's
        # last step size.
        fwd_t = current_time(sol)
        dt0 = length(fwd_t) > 1 ?
            sign(tspan[2] - tspan[1]) * abs(fwd_t[end] - fwd_t[end - 1]) :
            (tspan[2] - tspan[1]) / 1000
        DAEProblem(
            daefun, dz0, z0, tspan, p;
            differential_vars = diffvars_aug, callback = cb, initializealg,
            dt = dt0
        )
    else
        odefun = ODEFunction{true, true}(sense, mass_matrix = Diagonal(mmdiag))
        ODEProblem(odefun, z0, tspan, p; callback = cb, initializealg)
    end
    if RetCB
        return adj_prob, rcb
    else
        return adj_prob
    end
end

function DAEAdjointProblem(sol, sensealg::AbstractAdjointSensitivityAlgorithm, args...; kwargs...)
    error(
        """
        `$(nameof(typeof(sensealg)))` is not currently supported for adjoint sensitivity
        analysis of fully implicit `DAEProblem`s. Use `InterpolatingAdjoint`, e.g.

            adjoint_sensitivities(sol, alg; sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()), ...)

        Alternatively, express the DAE in mass-matrix form as an `ODEProblem`
        (e.g. via ModelingToolkit), for which all continuous adjoint methods are
        supported.
        """
    )
end
