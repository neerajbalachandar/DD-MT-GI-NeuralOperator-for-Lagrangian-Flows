#=############################################################################## # DESCRIPTIO
    Driver of round jet simulations.

# AUTHORSHIP
  * Author    : Eduardo J. Alvarez
  * Email     : Edo.AlvarezR@gmail.com
  * Created   : Sep 2021
  * Copyright : Eduardo J. Alvarez. All rights reserved.
=###############################################################################

# FLOWVPM's installed zeta_fmm calls FastMultipole.Tree without the explicit
# source/target flag required by the FastMultipole version in the FLOWUnsteady
# environment. Keep the same direct-list accumulation while supplying it.
function zeta_fmm_compat(pfield; progress=false, deadline=Inf)
    vpm.get_np(pfield) == 0 && return nothing
    started = time()
    time() < deadline || error("Roundjet RBF exceeded its configured time limit before an FMM evaluation")
    if progress
        println("[roundjet RBF] building FMM interaction tree for $(vpm.get_np(pfield)) particles")
        flush(stdout)
    end
    options = pfield.fmm
    # FastMultipole 1.x expects a one-system static vector here; the installed
    # FLOWVPM zeta_fmm passes its scalar ncrit and fails during tree subdivision.
    leaf_size = vpm.FastMultipole.default_leaf_size((pfield,))
    tree = vpm.FastMultipole.Tree((pfield,), false; leaf_size,
        shrink_recenter=options.shrink_recenter)
    _, direct_list = vpm.FastMultipole.build_interaction_lists(
        tree.branches, tree.branches, leaf_size, options.theta, false, true, true)
    time() < deadline || error("Roundjet RBF exceeded its configured time limit while building the FMM interaction tree")
    sort_index = tree.sort_index_list[1]
    zeta = pfield.kernel.zeta
    branches = tree.branches

    # Direct-list entries for one source leaf all accumulate into particles in
    # that leaf. Group them so each threaded worker owns disjoint output
    # particles and the vorticity sums need no atomics.
    targets_by_source = [Int[] for _ in eachindex(branches)]
    for (i_target, i_source) in direct_list
        push!(targets_by_source[i_source], i_target)
    end
    active_sources = findall(targets -> !isempty(targets), targets_by_source)
    if progress
        println("[roundjet RBF] FMM evaluation: $(vpm.get_np(pfield)) particles, $(length(direct_list)) near-field leaf pairs, $(Threads.nthreads()) threads")
        flush(stdout)
    end

    isempty(active_sources) && return nothing
    completed_sources = Threads.Atomic{Int}(0)
    report_every = max(1, cld(length(active_sources), 10))
    Threads.@threads for work_index in eachindex(active_sources)
        i_source = active_sources[work_index]
        source_branch = branches[i_source]
        for i_target in targets_by_source[i_source]
            target_branch = branches[i_target]
            for i_source_particle in source_branch.bodies_index[1]
                Pi = vpm.get_particle(pfield, sort_index[i_source_particle])
                X_i = vpm.get_X(Pi)
                J = vpm.get_J(Pi)
                for i_target_particle in target_branch.bodies_index[1]
                    Pj = vpm.get_particle(pfield, sort_index[i_target_particle])
                    X_j = vpm.get_X(Pj)
                    dx = X_i[1] - X_j[1]
                    dy = X_i[2] - X_j[2]
                    dz = X_i[3] - X_j[3]
                    sigma = vpm.get_sigma(Pj)[]
                    zeta_sigma = zeta(sqrt(dx*dx + dy*dy + dz*dz) / sigma) / sigma^3
                    Gamma = vpm.get_Gamma(Pj)
                    for k in 1:3
                        J[k] += Gamma[k] * zeta_sigma
                    end
                end
            end
        end
        if progress
            completed = Threads.atomic_add!(completed_sources, 1) + 1
            if completed == 1 || completed % report_every == 0 || completed == length(active_sources)
                elapsed = round(time() - started; digits=1)
                println("[roundjet RBF] FMM progress: $completed/$(length(active_sources)) source leaves ($(round(Int, 100*completed/length(active_sources)))%), $(elapsed)s")
                flush(stdout)
            end
            time() < deadline || error("Roundjet RBF exceeded its configured time limit during an FMM evaluation")
        end
    end
    if progress
        println("[roundjet RBF] FMM evaluation completed in $(round(time() - started; digits=1))s")
        flush(stdout)
    end
    return nothing
end


function run_roundjet_simulation(pfield::vpm.ParticleField,
                                        # ------- SIMULATION PARAMETERS --------
                                        d::Real,        # (m) jet diameter
                                        U1::Real,       # (m/s) Inflow centerline velocity
                                        Vfreestream::Array{<:Real, 1}; # (m/s) freestream vector
                                        AR=1.0,         # Aspect ratio a/b of jet NOTE: DON'T USE ANYTHING BUT 1.0 WITH rbf==true, AS IT HAS BEEN HARDCODED TO BE ROUNDED
                                        O=zeros(3),     # Origin of jet
                                        Oaxis=Float64[i==j for i in 1:3, j in 1:3], # Orientation of jet
                                        thetaod=0.025,  # Ratio of inflow momentum thickness of shear layer to diameter, θ/d
                                        Vprofile=(r, d, theta) -> abs(r) < d/2 ? tanh( (d/2 - abs(r)) / (theta) ) : 0, # Velocity profile of at boundary condition
                                        Re=nothing,     # Reynolds number
                                        # ------- SOLVER OPTIONS ---------------
                                        steps_per_d=50, # Number of time steps for the centerline at U1 to travel one diameter
                                        d_travel_tot=10,# Run simulation for an equivalent of this many diameters
                                        pulse_steps=10, # Finite inlet pulse length in integration steps; no runtime shedding.
                                        dxotheta=1/4,   # Distance Δx between particles over momentum thickness θ
                                        overlap=2.4,    # Overlap between particles
                                        minWfraction=1e-2,  # Threshold at which not to add particles
                                        maxRoR=1.0,     # (m) maximum radial distance to discretize
                                        rbf=true,       # If true, it runs an RBF interpolation to match the analytic vorticity with more accuracy
                                        rbf_optargs=[(:itmax,200), (:tol,2.5e-2), (:iterror,true), (:verbose,true), (:debug,false)],
                                        rbf_timeout_seconds=1800.0,
                                        rbf_Wf=0.5,     # Scales target RBF vorticity by this factor
                                        rbf_fullcylinder=false,  # Use a cylinder of particles for the RBF with half of it in inside the jet
                                        keep_rbfparticles=false, # Keep particles used for RBF as static particles
                                        keep_rbffullcylinder=false, # Keep the half of the cylinder that is inside the jet as free particles
                                        restart_file=nothing,
                                        restart_sigma=nothing,
                                        # ------- OUTPUT OPTIONS ---------------
                                        save_path=nothing,
                                        prompt=true,            # Whether to prompt the user
                                        verbose=true,           # Enable verbose
                                        v_lvl=0,
                                        runtime_functions=[],   # Monitors and runtime functions
                                        run_name="roundjet",    # Prefix of outputfiles
                                        optargs...
                                        )

    # -------------- SIMULATION PARAMETERS -------------------------------------
    # Jet geometry
    R         = d/2                         # (m) jet radius
    Cline     = Oaxis[:, 3]                 # Centerline direction

    # Temporal discretization
    dt         = d/steps_per_d / U1         # (s) time step
    nsteps     = ceil(Int, d_travel_tot*d / U1 / dt) # Number of time steps
    pulse_steps >= 1 || error("pulse_steps must be at least one")
    pulse_steps < nsteps || error("pulse_steps must be shorter than the roundjet trajectory")

    # Spatial discretization
    maxR       = maxRoR*R
    dx         = dxotheta * thetaod * d     # (m) approximate distance between particles
    sigma      = overlap*dx                 # particle smoothing

    # Inflow jet velocity (not including freestream)
    Vjet(r) = U1*Vprofile(r, d, thetaod*d)

    # -------------- SIMULATION SETUP ------------------------------------------

    # Convert velocity profile into vorticity profile
    Vjet_wrap(X) = Vjet(X[1])
    g(X) = ForwardDiff.gradient(Vjet_wrap, X)
    dVdr(r) = g([r])[1]
    W(r) = -dVdr(r)

    # Brute-force find maximum vorticity in the region to discretize
    Wpeak = maximum([abs.(W.(r)) for r in range(-maxR, maxR, length=1000)])

    # Pre-calculations for discretization
    Nphi = ceil(Int, 2*pi*R/dx)             # Number of cross sections
    NR = ceil(Int, maxR/dx)                 # Number of radial sections (annuli)
    dr = maxR/NR                            # (m) actual radial distance between particles

    if verbose
        println("\t"^(v_lvl+1)*"Number of cross-sections Nphi:\t$(Nphi)")
        println("\t"^(v_lvl+1)*"Number of annuli NR:\t\t$(NR)")
    end

    V2 = dot(Vfreestream, Cline)     # Axial component of the freestream

    # Spatial discretization of the finite jet pulse
    for ri in 1:NR         # Iterate over annuli (NOTE: Here we assume that the profile is symmetric from -R to R)

        # Lower and upper bounds, and center of annulus
        rlo = dr*(ri-1)                     # Annulus lower bound
        rup = dr*ri                         # Annulus uper bound
        rc = (rlo+rup)/2                    # Annulus center

        Vc = V2 + Vjet(rc)                  # Velocity at center of annulus
        dz = Vc*dt                          # Distance traveled in one time step

        # Integrate vorticity radially over annulus segment
        # NOTE: This line implicitely assumes AR=1.0
        Wint, err = HCubature.hquadrature(W, rlo, rup; rtol=1e-8, atol=0, maxevals=1000)

        circulation = Wint*dz + 1e-12       # Annulus circulation
        Wmean = Wint / (rup-rlo)            # Mean vorticity

        # Longitudinal area (in z-r plane) of annulus swept by the velocity
        # NOTE: This line implicitely assumes AR=1.0
        area = dz*(rup-rlo)

        if abs(Wmean) / Wpeak >= minWfraction
            # Initialize the entire finite-duration jet pulse at t=0. Every
            # particle is active from the start, so particle count stays
            # constant instead of growing by one inlet layer each timestep.
            for zi in 0:pulse_steps

                for sgn in [1, -1][1:2^(rbf_fullcylinder && zi!=0)]
                    this_O = O + sgn*zi*dz*Cline  # Center of this layer

                    # Discretize the annulus into Nphi sections as particles
                    addannulus(pfield, circulation,
                                            rc, AR,
                                            Nphi, sigma, area;
                                            O=this_O, Oaxis=Oaxis,
                                            static=false,
                                            verbose=verbose, v_lvl=v_lvl+1)
                end
            end
        end
    end

    if verbose
        println("\t"^v_lvl*"Initialized finite-pulse particles:\t$(vpm.get_np(pfield))")
    end

    # Determine the vortex strength of the boundary condition through RBF on the target vorticity
    if rbf

        if verbose
            println("\t"^v_lvl*"RBF fit to the analytical profile...")
        end

        Wp = zeros(3)

        # Use analytic vorticity as target vorticity (stored under P.M[7:9])
        for P in vpm.iterate(pfield; include_static=true)
            X = vpm.get_X(P)
            Gamma = vpm.get_Gamma(P)
            M = vpm.get_M(P)

            # Radial position of this particle
            aux = dot(X, Cline)
            r = 0
            for i in 1:3; r += (X[i] - aux*Cline[i])^2; end
            r = sqrt(r)

            # Vorticity at this particle
            Wp .= Gamma
            Wp .*= W(r) / norm(Gamma)

            for i in 1:3
                M[i+6] = Wp[i]

                # NOTE: If I don't take only half of the target vorticity
                #       the simulation ends up having double the velocity
                #       and vorticity once the jet is fully developed
                M[i+6] *= rbf_Wf

            end
        end

        # Keep all finite-pulse particles active during the RBF fit.
        for P in vpm.iterate(pfield; include_static=true); vpm.set_static(P, false); end

        # RBF interpolation of the analytic distribution
        rbf_timeout_seconds > 0 || error("rbf_timeout_seconds must be positive")
        rbf_started = time()
        rbf_deadline = rbf_started + rbf_timeout_seconds
        rbf_evaluations = Ref(0)
        println("[roundjet RBF] starting conjugate-gradient fit: $(vpm.get_np(pfield)) particles, max $(get(Dict(rbf_optargs), :itmax, 200)) iterations, timeout $(rbf_timeout_seconds)s")
        flush(stdout)
        function monitored_rbf_zeta(field)
            rbf_evaluations[] += 1
            println("[roundjet RBF] operator evaluation $(rbf_evaluations[]) starting; $(vpm.get_np(field)) particles")
            flush(stdout)
            return zeta_fmm_compat(field; progress=true, deadline=rbf_deadline)
        end
        rbf_scheme = vpm.CoreSpreading(-1, -1, monitored_rbf_zeta; v_lvl=v_lvl+1, rbf_optargs...)
        vpm.rbf_conjugategradient(pfield, rbf_scheme)
        println("[roundjet RBF] fit completed in $(round(time() - rbf_started; digits=1))s using $(rbf_evaluations[]) operator evaluations")

        # This is a finite pulse, not an inlet boundary condition: all fitted
        # particles remain active for the full trajectory.
        for P in vpm.iterate(pfield; include_static=true); vpm.set_static(P, false); end
    else
        # The annulus strengths already integrate the analytic vorticity
        # profile over each particle's support. Keep that discretization and
        # apply the same calibration used by the RBF target without solving a
        # global, quadratic-cost conjugate-gradient problem.
        for P in vpm.iterate(pfield; include_static=true)
            vpm.get_Gamma(P) .*= rbf_Wf
        end
        if verbose
            println("[roundjet] using analytically integrated annulus strengths (scale=$(rbf_Wf)); iterative RBF fit disabled")
            flush(stdout)
        end
    end

    vpm.get_np(pfield) <= pfield.maxparticles || error("Finite roundjet pulse exceeds its preallocated particle capacity")
    if verbose
        println("\t"^v_lvl*"Finite-pulse particles:\t$(vpm.get_np(pfield))")
        println("[roundjet] initialized a $(pulse_steps/steps_per_d)D finite pulse; particle count is fixed")
        println("[roundjet] particle buffer capacity=$(pfield.maxparticles)")
        flush(stdout)
    end

    # Simulation restart
    if restart_file != nothing
        # Read restart file, overwritting the particle field
        vpm.read!(pfield, restart_file; overwrite=true, load_time=true)

        if restart_sigma != nothing

            # Evaluate current vorticity field (gets stored under get_J(P)[1:3])
            zeta_fmm_compat(pfield)

            # Resize particle cores and store target vorticity under P.M[7:9]
            for P in vpm.iterate(pfield)
                vpm.get_sigma(P)[1] = restart_sigma

                for i in 1:3
                    vpm.get_M(P)[i+6] = vpm.get_J(P)[i]
                end
            end

            # Calculate the new vortex strenghts through RBF
            rbf_scheme = vpm.CoreSpreading(-1, -1, zeta_fmm_compat; v_lvl=v_lvl+1, rbf_optargs...)
            vpm.rbf_conjugategradient(pfield, rbf_scheme)

        end
    end

    # Overwrite kinematic viscosity with the requested Reynolds number
    if Re != nothing && vpm.isinviscid(pfield.viscous) == false

        # (m^2/s) kinematic viscosity. Note: Is this correct?
        pfield.viscous.nu = (U1 - V2) * d / Re

        if vpm.iscorespreading(pfield.viscous)
            # Set core size for reset after overgrowing
            pfield.viscous.sgm0 = sigma
        end
    end

    # --------------- DEFINE RUNTIME FUNCTION ----------------------------------
    expected_particles = vpm.get_np(pfield)
    function check_fixed_particle_count(field, t, dt; optargs...)
        actual = vpm.get_np(field)
        actual == expected_particles || error("Roundjet particle count changed from $expected_particles to $actual at t=$t")
        return false
    end
    this_runtime_functions = vcat(check_fixed_particle_count, runtime_functions)
    runtime_function(args...; optargs...) = !prod(!f(args...; optargs...) for f in this_runtime_functions)


    # -------------- SIMULATION ------------------------------------------------
    vpm.run_vpm!(pfield, dt, nsteps;    runtime_function=runtime_function,
                                        save_path=save_path,
                                        run_name=run_name,
                                        prompt=prompt,
                                        verbose=verbose, v_lvl=v_lvl,
                                        optargs...
                                        )
    return pfield
end



function run_roundjet_simulation(d::Real, U1::Real,
                                        U2::Real,           # (m/s) Coflow velocity
                                        args...;
                                        U2angle=[0,0,0],    # (deg) Coflow angle from centerline
                                        Oaxis=Float64[i==j for i in 1:3, j in 1:3],
                                        maxparticles::Int=Int(3e6), pfieldargs=(),
                                        optargs...)

    # Define freestream (coflow) velocity
    Vfreestream = U2 * gt.rotation_matrix2(U2angle...) * Oaxis[:, 3]
    Vinf(t)     = Vfreestream

    # Start particle field with the target maximum number of particles
    pfield = vpm.ParticleField(maxparticles; Uinf=Vinf, pfieldargs...)

    return run_roundjet_simulation(pfield, d, U1, Vfreestream, args...;
                                            Oaxis=Oaxis,
                                            optargs...
                                            )
end
