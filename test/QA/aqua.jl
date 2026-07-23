using SciMLTesting, SciMLSensitivity, SciMLBase, Test
using Pkg

run_qa(
    SciMLSensitivity;
    explicit_imports = true,
    aqua_kwargs = (;
        ambiguities = (; recursive = false),
        # The persistent-tasks check is run separately below with a retry on the
        # retryable cold-cache parallel-precompile failure (SciML/SciMLSensitivity#1554,
        # JuliaTesting/Aqua.jl#315). Disable Aqua's single-shot version here so it does
        # not misclassify that flake as a persistent task.
        persistent_tasks = false,
        piracies = (;
            treat_as_own = [
                SciMLBase._concrete_solve_adjoint,
                SciMLBase._concrete_solve_forward,
            ],
        ),
    ),
    ei_kwargs = (;
        # `BrownFullBasicInit`/`DefaultInit` are owned by DiffEqBase but re-exported
        # through OrdinaryDiffEqCore, which is where SciMLSensitivity imports them.
        # The rest are re-imported through the parent `SciMLSensitivity` by the
        # Mooncake extension (`using SciMLSensitivity: ...`), an intentional idiom.
        all_explicit_imports_via_owners = (;
            ignore = (
                :BrownFullBasicInit, :DefaultInit,
                # SciMLSensitivityMooncakeExt: re-imported through the parent module
                :DiffEqBase, :FunctionWrappersWrappers, :ODEFunction, :SciMLBase,
                :SciMLStructures, :Tunable, :canonicalize, :current_time,
                :isscimlstructure, :state_values, :unwrapped_f,
            ),
        ),
        # Non-public names of upstream deps imported explicitly here; ignore until
        # those packages mark them public (each grouped by its source package).
        all_explicit_imports_are_public = (;
            ignore = (
                # ChainRulesCore
                :AbstractTangent,
                # OrdinaryDiffEqCore
                :BrownFullBasicInit, :DefaultInit, :default_nlsolve, :has_autodiff,
                # SciMLBase
                :AbstractAdjointSensitivityAlgorithm,
                :AbstractForwardSensitivityAlgorithm, :AbstractOptimizationProblem,
                :AbstractOverloadingSensitivityAlgorithm,
                :AbstractSecondOrderSensitivityAlgorithm, :AbstractSensitivityAlgorithm,
                :AbstractShadowingSensitivityAlgorithm, :AbstractTimeseriesSolution,
                :unwrapped_f,
                # SciMLStructures
                :Tunable, :canonicalize, :isscimlstructure,
                # SciMLSensitivityMooncakeExt re-imports these internal/non-public
                # names through the parent module (`using/import SciMLSensitivity: ...`),
                # the intentional extension idiom. They are SciMLSensitivity internals
                # or deps re-exported by the parent, so they are not public in
                # SciMLSensitivity and never will be.
                :DiffEqBase, :FakeIntegrator, :FunctionWrappersWrappers, :MooncakeLoaded,
                :MooncakeVJP, :ODEFunction, :SciMLBase, :SciMLStructures,
                :SciMLStructuresCompatibilityError, :_init_originator_gradient,
                :convert_tspan, :current_time, :get_cb_paramjac_config,
                :get_paramjac_config, :has_continuous_callback, :mooncake_run_ad,
                :state_values,
            ),
        ),
        # Non-public names of upstream deps accessed qualified in the source; ignore
        # until those packages mark them public (each grouped by its source package).
        all_qualified_accesses_are_public = (;
            ignore = (
                # ArrayInterface
                :parameterless_type,
                # Base
                :(var"@pure"), :_nt_names, :diff_names,
                # DiffEqCallbacks
                :PeriodicCallbackAffect,
                # DiffEqNoiseProcess
                :vec_NoiseProcess,
                # Enzyme
                :EnzymeCore,
                # EnzymeCore
                :Mode,
                # EnzymeCore.EnzymeRules
                :inactive_type, :inactive,
                # FiniteDiff
                :DerivativeCache, :GradientCache, :JacobianCache,
                :finite_difference_derivative!, :finite_difference_gradient!,
                :finite_difference_jacobian, :finite_difference_jacobian!,
                # ForwardDiff
                :Chunk, :DerivativeConfig, :Dual, :GradientConfig, :JacobianConfig,
                :Partials, :Tag, :construct_seeds, :derivative!, :gradient!, :jacobian,
                :jacobian!, :npartials, :partials, :pickchunksize, :value,
                # LinearSolve
                :needs_concrete_A,
                # Mooncake (internal tangent/rrule API used by SciMLSensitivityMooncakeExt)
                :CoDual, :NoFData, :Tangent, :build_rrule, :tangent_to_primal!!,
                :zero_rdata,
                # OrdinaryDiffEqCore
                :alg_autodiff, :default_linear_interpolation,
                # ReverseDiff
                :GradientTape, :TrackedArray, :compile, :deriv, :forward_pass!, :gradient,
                :increment_deriv!, :input_hook, :output_hook, :pull_value!, :reverse_pass!,
                :unseed!, :value, :value!,
                # SciMLBase
                :ADOriginator, :AbstractDAESolution, :AbstractRODEProblem,
                :AbstractSDDEProblem,
                :AlgorithmInterpretation, :ChainRulesOriginator, :EnzymeOriginator,
                :FullSpecialize, :ImmutableNonlinearProblem, :MooncakeOriginator,
                :OVERDETERMINED, :ParamJacobianWrapper, :ReverseDiffOriginator,
                :TrackerOriginator, :UDerivativeWrapper, :UJacobianWrapper, :Void,
                :_concrete_solve_adjoint, :_concrete_solve_forward, :alg_interpretation,
                :enable_interpolation_sensitivitymode,
                :has_initialization_data, :has_observed, :has_paramjac, :has_vjp_p,
                :initialization_status, :sensitivity_solution, :specialization,
                # SciMLStructures
                :replace,
                # SparseArrays
                :AbstractSparseMatrixCSC,
                # Tracker
                :TrackedReal, :collect, :data, :forward,
                # Zygote
                :Buffer, :accum,
            ),
        ),
    ),
)

# Persistent-tasks check with a retry on the retryable cold-cache precompile failure.
#
# Aqua's built-in check (disabled above) spawns a throwaway wrapper package that
# `using`s SciMLSensitivity and, from inside the precompilation process, writes a
# `done.log` sentinel; a package that never lets that process exit within `tmax` is
# reported as holding a persistent `Task`. When the wrapper's freshly resolved
# dependency set precompiles cold and in parallel, one module can hit a *retryable*
# "missing from the cache" failure (observed here: `Accessors`; in the issue:
# `NonlinearSolveBaseChainRulesCoreExt` -- the culprit is scheduling-dependent).
# `Pkg.precompile` then exits without compiling the wrapper, so `done.log` is never
# written and the check misreports it as a persistent task
# (SciML/SciMLSensitivity#1554, JuliaTesting/Aqua.jl#315). An immediate second
# precompile finds the now-warm cache and succeeds.
#
# This faithfully reproduces Aqua's wrapper (its public API only exposes the
# single-shot `Aqua.test_persistent_tasks`) but retries when the child precompile
# exits without writing `done.log`. A genuine persistent task is unaffected: it
# writes `done.log` and then keeps the process alive past `tmax`, which this still
# detects, so the retry cannot mask a real background task.
function has_persistent_tasks_with_retry(package::Module; tmax = 10, retries = 3)
    pkgname = string(nameof(package))
    pkgpath = pkgdir(package)
    prev_project = Base.active_project()::String
    isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(false)
    try
        for attempt in 1:retries
            wrapperdir = tempname()
            wrappername, _ = only(Pkg.generate(wrapperdir; io = devnull))
            Pkg.activate(wrapperdir; io = devnull)
            Pkg.develop(Pkg.PackageSpec(path = pkgpath); io = devnull)
            statusfile = joinpath(wrapperdir, "done.log")
            open(joinpath(wrapperdir, "src", wrappername * ".jl"), "w") do io
                println(
                    io,
                    """
                    module $wrappername
                    using $pkgname
                    open("$(escape_string(statusfile))", "w") do io
                        println(io, "done")
                        flush(io)
                    end
                    end
                    """,
                )
            end
            cmd = `$(Base.julia_cmd()) --project=$wrapperdir -e 'push!(LOAD_PATH, "@stdlib"); using Pkg; Pkg.precompile()'`
            proc = run(cmd, stdin, stdout, stderr; wait = false)
            while !isfile(statusfile) && process_running(proc)
                sleep(0.5)
            end
            if !isfile(statusfile)
                # Child precompile exited without loading the wrapper: the retryable
                # cold-cache failure. Retry with a fresh wrapper; the next precompile
                # hits the now-warm cache. Only give up after `retries` attempts.
                if attempt < retries
                    @info "Persistent-tasks wrapper precompile exited without writing done.log; retrying" attempt retries
                    continue
                end
                @error "Persistent-tasks wrapper precompile never produced done.log after $retries attempts"
                return true
            end
            t = time()
            while process_running(proc) && time() - t < tmax
                sleep(0.1)
            end
            exited = !process_running(proc)
            exited || kill(proc, Base.SIGKILL)
            return !exited
        end
        return true
    finally
        isdefined(Pkg, :respect_sysimage_versions) && Pkg.respect_sysimage_versions(true)
        Pkg.activate(prev_project; io = devnull)
    end
end

@testset "Persistent tasks" begin
    with_clean_persistent_tasks_sources() do
        @test !has_persistent_tasks_with_retry(SciMLSensitivity)
    end
end
