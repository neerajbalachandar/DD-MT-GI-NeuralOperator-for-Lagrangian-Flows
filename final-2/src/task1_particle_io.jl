module Task1ParticleIO

using LinearAlgebra

export get_particle_state, set_particle_ugradu!, build_input_features

"""
Read particle state from FLOWVPM particle field.

NOTE: Field names can differ across FLOWVPM versions.
If one accessor fails, edit this file only (no package edits needed).
"""
function get_particle_state(pfield)
    X = copy(pfield.X)'                # (N,3)
    Gamma = copy(pfield.Gamma)'        # (N,3)
    sigma = copy(pfield.sigma)         # (N,)
    return X, Gamma, sigma
end

"""
Build feature matrix expected by Task-1-u/gradU model.
Default channels:
[x,y,z,Gamma_x,Gamma_y,Gamma_z,sigma]
Optional context channels can be appended in this function if model used them.
"""
function build_input_features(X, Gamma, sigma; use_context_channels=false, phase=0.0, aoa_deg=0.0, freestream=(0.0,0.0,0.0))
    N = size(X, 1)
    feats = hcat(X[:,1], X[:,2], X[:,3], Gamma[:,1], Gamma[:,2], Gamma[:,3], sigma)

    if use_context_channels
        ctx = hcat(fill(phase, N), fill(aoa_deg, N), fill(freestream[1], N), fill(freestream[2], N), fill(freestream[3], N))
        feats = hcat(feats, ctx)
    end

    return Array{Float32}(feats)
end

"""
Write predicted U and gradU back into pfield arrays.
Expected `Y` columns:
1:3  -> U
4:6  -> gradUx row
7:9  -> gradUy row
10:12-> gradUz row
"""
function set_particle_ugradu!(pfield, Y::AbstractMatrix)
    U = Y[:, 1:3]
    Gx = Y[:, 4:6]
    Gy = Y[:, 7:9]
    Gz = Y[:, 10:12]

    # Most common layout in FLOWVPM code paths:
    pfield.U .= U'
    pfield.J[1, :, :] .= Gx'
    pfield.J[2, :, :] .= Gy'
    pfield.J[3, :, :] .= Gz'

    return nothing
end

end # module
