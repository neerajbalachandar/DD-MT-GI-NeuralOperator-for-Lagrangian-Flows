module FlowUnsteadyExportUGradU

using JLD2
using Printf

export build_export_callback

"""
Create a callback that saves per-timestep particle state + supervision arrays.

Expected arrays per timestep:
- X            : particle positions (N,3)
- Gamma        : particle vector strength (N,3)
- sigma        : core size (N)
- velocity     : U on particles (N,3)
- gradU_x/y/z  : velocity-gradient rows (N,3) each

Adjust field-access lines to your FLOWVPM version if names differ.
"""
function build_export_callback(save_dir::String; stride::Int=1)
    mkpath(save_dir)

    function cb(sim, pfield, t, dt; optargs...)
        step = Int(round(t / dt))
        if step % stride != 0
            return true
        end

        # ---- Version-dependent access (edit if needed) ----
        # The following are placeholders matching common FLOWVPM conventions.
        X = copy(pfield.X)'              # -> (N,3)
        Gamma = copy(pfield.Gamma)'      # -> (N,3)
        sigma = copy(pfield.sigma)       # -> (N)
        U = copy(pfield.U)'              # -> (N,3)
        Gx = copy(pfield.J[1, :, :])'    # -> (N,3)
        Gy = copy(pfield.J[2, :, :])'
        Gz = copy(pfield.J[3, :, :])'

        fname = @sprintf("particle_supervision_%06d.jld2", step)
        fpath = joinpath(save_dir, fname)

        @save fpath X Gamma sigma U Gx Gy Gz t dt step
        return true
    end

    return cb
end

end # module
