module SciMLSensitivitySundialsExt

using SciMLSensitivity: SciMLSensitivity, SciMLBase,
    SundialsAdjoint, QuadratureAdjoint, GaussAdjoint,
    ODEQuadratureAdjointSensitivityFunction, GaussIntegrand,
    vecjacobian!, vec_pjac!, ReverseDiffVJP, mutable_zeros,
    canonicalize, Tunable, isscimlstructure, unwrapped_f
import SciMLSensitivity: _adjoint_sensitivities
using SciMLBase: ODEProblem, isinplace
using Sundials: Sundials, N_Vector, NVector, realtype
using LinearAlgebra: I, UniformScaling

_cvodes_lmm(::Sundials.CVODE_BDF) = Sundials.CV_BDF
_cvodes_lmm(::Sundials.CVODE_Adams) = Sundials.CV_ADAMS
_cvodes_method(::Sundials.SundialsODEAlgorithm{M, LS}) where {M, LS} = M
_cvodes_linear_solver(::Sundials.SundialsODEAlgorithm{M, LS}) where {M, LS} = LS

function _check_cvodes_flag(flag, fname)
    return flag >= Sundials.CV_SUCCESS ||
        error("SundialsAdjoint: $fname failed with error code = $flag")
end

# User data passed through the CVODES C callbacks via `CVodeSetUserData(B)`.
# `S` carries the state vjp machinery (`vecjacobian!`) and `integrand` the
# parameter vjp machinery (`vec_pjac!`); both are the standard SciMLSensitivity
# caches so all `autojacvec` choices work unchanged.
mutable struct CVODESAdjointUserData{ND, F, P, S, GI, DL, PQ}
    const f::F
    const p::P
    const sz::NTuple{ND, Int}
    const n::Int
    const S::S
    const integrand::GI
    const dλ::DL
    const pquad::PQ
end

function _cvodes_forward_rhs(
        t::realtype, y_nv::N_Vector, ydot_nv::N_Vector,
        data::CVODESAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(y_nv), data.sz)
    dy = unsafe_wrap(
        Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(ydot_nv),
        data.sz
    )
    data.f(dy, y, data.p, t)
    return Sundials.CV_SUCCESS
end

# yB' = -(df/du)^T yB, evaluated at the CVODES checkpoint-interpolated forward
# state `y` which the backward integrator hands to this callback.
function _cvodes_adjoint_rhs(
        t::realtype, y_nv::N_Vector, yB_nv::N_Vector,
        yBdot_nv::N_Vector, data::CVODESAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(y_nv), data.sz)
    λ = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yB_nv), data.n)
    out = unsafe_wrap(
        Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yBdot_nv),
        data.n
    )
    vecjacobian!(data.dλ, y, λ, data.p, t, data.S)
    @. out = -data.dλ
    return Sundials.CV_SUCCESS
end

# qB' = -(df/dp)^T yB. CVODES integrates qB backward from tf with qB(tf) = 0,
# so the value returned at t0 is +∫_{t0}^{tf} λ^T (df/dp) dt.
function _cvodes_quad_rhs(
        t::realtype, y_nv::N_Vector, yB_nv::N_Vector,
        qBdot_nv::N_Vector, data::CVODESAdjointUserData{ND}
    ) where {ND}
    y = unsafe_wrap(Array{Float64, ND}, Sundials.N_VGetArrayPointer_Serial(y_nv), data.sz)
    λ = unsafe_wrap(Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(yB_nv), data.n)
    out = unsafe_wrap(
        Vector{Float64}, Sundials.N_VGetArrayPointer_Serial(qBdot_nv),
        length(data.pquad)
    )
    vec_pjac!(data.pquad, λ, y, t, data.integrand)
    pq = vec(data.pquad)
    @. out = -pq
    return Sundials.CV_SUCCESS
end

function _forward_cfunction(::T) where {T <: CVODESAdjointUserData}
    return @cfunction(
        _cvodes_forward_rhs, Cint,
        (realtype, N_Vector, N_Vector, Ref{T})
    )
end

function _adjoint_cfunction(::T) where {T <: CVODESAdjointUserData}
    return @cfunction(
        _cvodes_adjoint_rhs, Cint,
        (realtype, N_Vector, N_Vector, N_Vector, Ref{T})
    )
end

function _quad_cfunction(::T) where {T <: CVODESAdjointUserData}
    return @cfunction(
        _cvodes_quad_rhs, Cint,
        (realtype, N_Vector, N_Vector, N_Vector, Ref{T})
    )
end

# Attaches the nonlinear/linear solver combination implied by `alg` via the
# supplied setter closures and returns the created objects so the caller can
# keep them rooted for the duration of the integration.
function _cvodes_set_solvers(ls_setter, nls_setter, alg, unv, n, ctx)
    linear_solver = _cvodes_linear_solver(alg)
    if _cvodes_method(alg) == :Functional
        nls = Sundials.SUNNonlinSol_FixedPoint(unv, Cint(0), ctx)
        nls_handle = Sundials.NonLinSolHandle(nls, Sundials.FixedPoint())
        _check_cvodes_flag(nls_setter(nls), "CVodeSetNonlinearSolver")
        return nls_handle, nothing, nothing
    elseif linear_solver == :Dense
        A = Sundials.SUNDenseMatrix(n, n, ctx)
        LS = Sundials.SUNLinSol_Dense(unv, A, ctx)
        _check_cvodes_flag(ls_setter(LS, A), "CVodeSetLinearSolver")
        return nothing, LS, A
    elseif linear_solver == :Band
        A = Sundials.SUNBandMatrix(n, alg.jac_upper, alg.jac_lower, ctx)
        LS = Sundials.SUNLinSol_Band(unv, A, ctx)
        _check_cvodes_flag(ls_setter(LS, A), "CVodeSetLinearSolver")
        return nothing, LS, A
    elseif linear_solver == :GMRES
        krylov_dim = alg.krylov_dim == 0 ? 5 : alg.krylov_dim
        LS = Sundials.SUNLinSol_SPGMR(unv, Cint(Sundials.PREC_NONE), Cint(krylov_dim), ctx)
        # A typed null pointer is required for dispatch to reach the ccall method.
        nullmat = Sundials.SUNMatrix(C_NULL)
        _check_cvodes_flag(ls_setter(LS, nullmat), "CVodeSetLinearSolver")
        return nothing, LS, nothing
    else
        error(
            "SundialsAdjoint currently supports the `:Dense`, `:Band`, and `:GMRES` " *
                "linear solvers (or `method = :Functional`), got `$(linear_solver)`."
        )
    end
end

function _adjoint_sensitivities(
        sol, sensealg::SundialsAdjoint{CS, AD, FDT},
        alg::Union{Sundials.CVODE_BDF, Sundials.CVODE_Adams};
        t = nothing,
        dgdu_discrete = nothing,
        dgdp_discrete = nothing,
        dgdu_continuous = nothing,
        dgdp_continuous = nothing,
        g = nothing, no_start = false,
        abstol = 1.0e-6, reltol = 1.0e-3,
        maxiters = Int(1.0e5),
        kwargs...
    ) where {CS, AD, FDT}
    if dgdu_continuous !== nothing || dgdp_continuous !== nothing || g !== nothing
        error(
            "SundialsAdjoint currently only supports discrete cost functionals " *
                "(`t` together with `dgdu_discrete`/`dgdp_discrete`). Continuous cost " *
                "functionals (`g`, `dgdu_continuous`, `dgdp_continuous`) are not " *
                "supported; use `GaussAdjoint` or `QuadratureAdjoint` for those."
        )
    end
    (t === nothing || dgdu_discrete === nothing) &&
        error("SundialsAdjoint requires `t` and `dgdu_discrete` to be specified.")

    prob = sol.prob
    prob isa ODEProblem ||
        error("SundialsAdjoint only supports `ODEProblem`s.")
    isinplace(prob) ||
        error("SundialsAdjoint currently only supports in-place (mutating) `ODEProblem`s.")
    mm = prob.f.mass_matrix
    (mm isa UniformScaling && mm == I) ||
        error("SundialsAdjoint does not support mass matrices (a CVODES restriction).")
    u0 = prob.u0
    eltype(u0) === Float64 && eltype(prob.tspan) === Float64 ||
        error("SundialsAdjoint requires `Float64` state and time (a SUNDIALS restriction).")
    t0, tf = prob.tspan
    t0 < tf || error("SundialsAdjoint requires a forward time span with `tspan[1] < tspan[2]`.")

    ts = collect(Float64, t)
    issorted(ts) || error("SundialsAdjoint requires the cost times `t` to be sorted in ascending order.")
    (first(ts) >= t0 && last(ts) <= tf) ||
        error("SundialsAdjoint requires all cost times `t` to lie within the problem `tspan`.")

    p = prob.p
    has_p = !(p === nothing || p isa SciMLBase.NullParameters)
    if has_p
        if isscimlstructure(p) && !(p isa AbstractArray)
            tunables, _, _ = canonicalize(Tunable(), p)
        elseif p isa AbstractArray
            tunables = p
        else
            error(
                "SundialsAdjoint requires the parameters to be an `AbstractArray` or a " *
                    "SciMLStructures-compatible struct, got `$(typeof(p))`."
            )
        end
    else
        tunables = nothing
    end

    vjp = sensealg.autojacvec === nothing ? ReverseDiffVJP() : sensealg.autojacvec
    vjp === true &&
        error(
        "SundialsAdjoint does not support `autojacvec = true`. Use `autojacvec = false` " *
            "(Jacobian construction controlled by `autodiff`) or a vjp choice such as " *
            "`ReverseDiffVJP()` or `EnzymeVJP()`."
    )

    # Reuse the existing quadrature/Gauss adjoint vjp caches: `S` computes the
    # state vjp `(df/du)^T λ` via `vecjacobian!` and `integrand` the parameter
    # vjp `(df/dp)^T λ` via `vec_pjac!`.
    S = ODEQuadratureAdjointSensitivityFunction(
        nothing,
        QuadratureAdjoint(
            chunk_size = CS, autodiff = AD, diff_type = FDT,
            autojacvec = vjp
        ),
        true, sol, nothing, nothing, alg
    )
    integrand = has_p ?
        GaussIntegrand(
            sol,
            GaussAdjoint(
                chunk_size = CS, autodiff = AD, diff_type = FDT,
                autojacvec = vjp
            ),
            nothing
        ) : nothing

    n = length(u0)
    np = has_p ? length(tunables) : 0
    f = unwrapped_f(prob.f)
    data = CVODESAdjointUserData(
        f, p, size(u0), n, S, integrand,
        zeros(n), has_p ? mutable_zeros(tunables) : nothing
    )
    fwd_cfun = _forward_cfunction(data)
    adj_cfun = _adjoint_cfunction(data)
    quad_cfun = has_p ? _quad_cfunction(data) : nothing

    interp = sensealg.interp === :hermite ? Sundials.CV_HERMITE : Sundials.CV_POLYNOMIAL
    lmm = _cvodes_lmm(alg)

    ctx_ref = Ref{Sundials.SUNContext}(C_NULL)
    Sundials.SUNContext_Create(
        C_NULL,
        Base.unsafe_convert(Ptr{Sundials.SUNContext}, ctx_ref)
    )
    ctx = ctx_ref[]
    mem = Sundials.Handle(Sundials.CVodeCreate(lmm, ctx))

    ufwd = collect(vec(copy(u0)))
    ufwd_nv = NVector(ufwd, ctx)
    yret = similar(ufwd)
    yret_nv = NVector(yret, ctx)
    λ = zeros(n)
    λ_nv = NVector(λ, ctx)
    qB = zeros(np)
    qB_nv = has_p ? NVector(qB, ctx) : nothing
    gu = zeros(n)
    gp = dgdp_discrete === nothing ? nothing : zeros(np)
    dp = has_p ? zeros(np) : nothing
    tret = [t0]
    ncheck = Ref{Cint}(0)
    which = Ref{Cint}(0)

    GC.@preserve data ufwd_nv yret_nv λ_nv qB_nv begin
        try
            # Forward pass with checkpointing.
            _check_cvodes_flag(
                Sundials.CVodeInit(mem, fwd_cfun, t0, ufwd_nv),
                "CVodeInit"
            )
            _check_cvodes_flag(Sundials.CVodeSetUserData(mem, data), "CVodeSetUserData")
            _check_cvodes_flag(
                Sundials.CVodeSStolerances(mem, reltol, abstol),
                "CVodeSStolerances"
            )
            _check_cvodes_flag(
                Sundials.CVodeSetMaxNumSteps(mem, maxiters),
                "CVodeSetMaxNumSteps"
            )
            nlsf, LSf,
                Af = _cvodes_set_solvers(
                (LS, A) -> Sundials.CVodeSetLinearSolver(mem, LS, A),
                NLS -> Sundials.CVodeSetNonlinearSolver(mem, NLS),
                alg, ufwd_nv, n, ctx
            )
            _check_cvodes_flag(
                Sundials.CVodeAdjInit(mem, sensealg.steps, interp),
                "CVodeAdjInit"
            )
            _check_cvodes_flag(
                Sundials.CVodeF(mem, tf, yret_nv, tret, Sundials.CV_NORMAL, ncheck),
                "CVodeF"
            )

            # Backward (adjoint) problem. λ(tf) collects the cost jumps at tf.
            cur_time = length(ts)
            if cur_time >= 1 && ts[cur_time] == tf
                y_f = sol(tf)
                while cur_time >= 1 && ts[cur_time] == tf
                    if !(no_start && cur_time == 1)
                        fill!(gu, false)
                        dgdu_discrete(gu, y_f, p, tf, cur_time)
                        λ .+= gu
                        if dgdp_discrete !== nothing
                            fill!(gp, false)
                            dgdp_discrete(gp, y_f, p, tf, cur_time)
                            dp .+= gp
                        end
                    end
                    cur_time -= 1
                end
            end

            _check_cvodes_flag(Sundials.CVodeCreateB(mem, lmm, which), "CVodeCreateB")
            _check_cvodes_flag(
                Sundials.CVodeInitB(mem, which[], adj_cfun, tf, λ_nv),
                "CVodeInitB"
            )
            _check_cvodes_flag(
                Sundials.CVodeSetUserDataB(mem, which[], data),
                "CVodeSetUserDataB"
            )
            _check_cvodes_flag(
                Sundials.CVodeSStolerancesB(mem, which[], reltol, abstol),
                "CVodeSStolerancesB"
            )
            _check_cvodes_flag(
                Sundials.CVodeSetMaxNumStepsB(mem, which[], maxiters),
                "CVodeSetMaxNumStepsB"
            )
            nlsB, LSB,
                AB = _cvodes_set_solvers(
                (LS, A) -> Sundials.CVodeSetLinearSolverB(mem, which[], LS, A),
                NLS -> Sundials.CVodeSetNonlinearSolverB(mem, which[], NLS),
                alg, λ_nv, n, ctx
            )
            if has_p
                _check_cvodes_flag(
                    Sundials.CVodeQuadInitB(mem, which[], quad_cfun, qB_nv),
                    "CVodeQuadInitB"
                )
                _check_cvodes_flag(
                    Sundials.CVodeQuadSStolerancesB(mem, which[], reltol, abstol),
                    "CVodeQuadSStolerancesB"
                )
                _check_cvodes_flag(
                    Sundials.CVodeSetQuadErrConB(mem, which[], 1),
                    "CVodeSetQuadErrConB"
                )
            end

            # Integrate backward, stopping at every discrete cost time to add the
            # jump `λ += dg/du` and reinitialize the backward problem.
            tcur = tf
            while cur_time >= 1
                s = ts[cur_time]
                if s < tcur && s > t0
                    _check_cvodes_flag(
                        Sundials.CVodeB(mem, s, Sundials.CV_NORMAL),
                        "CVodeB"
                    )
                    _check_cvodes_flag(
                        Sundials.CVodeGetB(mem, which[], tret, λ_nv),
                        "CVodeGetB"
                    )
                    if has_p
                        _check_cvodes_flag(
                            Sundials.CVodeGetQuadB(mem, which[], tret, qB_nv),
                            "CVodeGetQuadB"
                        )
                    end
                    tcur = s
                    y_s = sol(s)
                    while cur_time >= 1 && ts[cur_time] == s
                        if !(no_start && cur_time == 1)
                            fill!(gu, false)
                            dgdu_discrete(gu, y_s, p, s, cur_time)
                            λ .+= gu
                            if dgdp_discrete !== nothing
                                fill!(gp, false)
                                dgdp_discrete(gp, y_s, p, s, cur_time)
                                dp .+= gp
                            end
                        end
                        cur_time -= 1
                    end
                    _check_cvodes_flag(
                        Sundials.CVodeReInitB(mem, which[], s, λ_nv),
                        "CVodeReInitB"
                    )
                    if has_p
                        _check_cvodes_flag(
                            Sundials.CVodeQuadReInitB(mem, which[], qB_nv),
                            "CVodeQuadReInitB"
                        )
                    end
                else
                    break
                end
            end

            # Final leg down to t0 and any jump exactly at t0.
            if tcur > t0
                _check_cvodes_flag(
                    Sundials.CVodeB(mem, t0, Sundials.CV_NORMAL),
                    "CVodeB"
                )
                _check_cvodes_flag(
                    Sundials.CVodeGetB(mem, which[], tret, λ_nv),
                    "CVodeGetB"
                )
                if has_p
                    _check_cvodes_flag(
                        Sundials.CVodeGetQuadB(mem, which[], tret, qB_nv),
                        "CVodeGetQuadB"
                    )
                end
            end
            while cur_time >= 1
                s = ts[cur_time]
                s == t0 ||
                    error("SundialsAdjoint: internal error, unprocessed cost time $(s).")
                if !(no_start && cur_time == 1)
                    y_s = sol(s)
                    fill!(gu, false)
                    dgdu_discrete(gu, y_s, p, s, cur_time)
                    λ .+= gu
                    if dgdp_discrete !== nothing
                        fill!(gp, false)
                        dgdp_discrete(gp, y_s, p, s, cur_time)
                        dp .+= gp
                    end
                end
                cur_time -= 1
            end

            has_p && (dp .+= qB)
        finally
            empty!(mem)
            Sundials.SUNContext_Free(ctx)
        end
    end

    du0 = copy(λ)
    return du0, has_p ? dp' : nothing
end

end
