module FlowUnsteadyMLBridge

"""
Template bridge for replacing particle U/gradU query with ML predictions.

Concept:
1) Build input features from current particle state at timestep t.
2) Call Python model inference (PyCall/PythonCall/IPC).
3) Write predicted U and gradU back to fields expected by FLOWVPM.
4) Let FLOWUnsteady continue standard time integration.

This module is intentionally minimal and should be adapted to your installed API.
"""

export apply_ml_ugradu!

function apply_ml_ugradu!(pfield, predictor; phase=0.0, aoa=0.0, freestream=(0.0,0.0,0.0))
    # 1) Build features from pfield state
    # X  = ... ; Gamma = ... ; sigma = ...

    # 2) y_pred = predictor(X_features)  # expected shape (N,12)
    # Split into U, gradU rows

    # 3) Write to pfield arrays used by integrator
    # pfield.U = ...
    # pfield.J = ...

    return nothing
end

end # module
