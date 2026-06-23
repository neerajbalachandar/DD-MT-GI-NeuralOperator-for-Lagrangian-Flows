module Task1ParticleIO

using LinearAlgebra
using Statistics

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

function _simple_geometry_proxy(X)
    center = mapslices(median, X; dims=1)
    rel = X .- center
    dist = sqrt.(sum(rel .^ 2; dims=2))[:, 1]
    scale = reshape(max.(dist, eps(Float64)), :, 1)
    normal = rel ./ scale
    return dist, normal
end

"""
Build a named feature matrix expected by the Python predictor.

The default names match the processed Task-1/Task-2 channels:
x/y/z, Gamma, sigma, simple geometry proxy, angle_of_attack, phase.
"""
function build_input_features(
    X,
    Gamma,
    sigma;
    feature_names=[
        "x", "y", "z",
        "Gamma_x", "Gamma_y", "Gamma_z",
        "sigma",
        "geom_dist", "geom_nx", "geom_ny", "geom_nz",
        "angle_of_attack", "phase",
    ],
    use_context_channels=true,
    phase=0.0,
    aoa_deg=0.0,
    freestream=(0.0,0.0,0.0),
)
    N = size(X, 1)
    geom_dist, geom_n = _simple_geometry_proxy(X)
    columns = Dict{String,Vector{Float64}}(
        "x" => X[:, 1],
        "y" => X[:, 2],
        "z" => X[:, 3],
        "Gamma_x" => Gamma[:, 1],
        "Gamma_y" => Gamma[:, 2],
        "Gamma_z" => Gamma[:, 3],
        "sigma" => vec(sigma),
        "geom_dist" => geom_dist,
        "geom_nx" => geom_n[:, 1],
        "geom_ny" => geom_n[:, 2],
        "geom_nz" => geom_n[:, 3],
        "geom_body_near" => zeros(Float64, N),
        "angle_of_attack" => fill(use_context_channels ? Float64(aoa_deg) : 0.0, N),
        "phase" => fill(use_context_channels ? Float64(phase) : 0.0, N),
        "freestream_x" => fill(use_context_channels ? Float64(freestream[1]) : 0.0, N),
        "freestream_y" => fill(use_context_channels ? Float64(freestream[2]) : 0.0, N),
        "freestream_z" => fill(use_context_channels ? Float64(freestream[3]) : 0.0, N),
    )

    missing = [name for name in feature_names if !haskey(columns, name)]
    if !isempty(missing)
        error("Unsupported feature names for runtime predictor: $(missing)")
    end

    feats = hcat([columns[String(name)] for name in feature_names]...)
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
