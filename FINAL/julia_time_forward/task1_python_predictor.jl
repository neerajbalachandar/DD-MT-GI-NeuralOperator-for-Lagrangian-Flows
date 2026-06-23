module Task1PythonPredictor

using NPZ

export build_predictor

"""
Build Python-backed predictor closure:
  Y = predictor(Xfeatures)
where Xfeatures is Float32 matrix (N,Fin), Y is Float32 matrix (N,12).

This uses PythonCall and expects a Python helper module at
`final-2/src/task1_python_predictor.py`.
"""
function build_predictor(model_py_path::String, model_meta_npz::String; device::String="cpu")
    using PythonCall

    pymod = pyimport("task1_python_predictor")
    predictor_obj = pymod.Task1UGradUPredictor(model_py_path, model_meta_npz, device)

    function predictor(Xfeatures::AbstractMatrix)
        Y = predictor_obj.predict(Array{Float32}(Xfeatures))
        return Array{Float32}(Y)
    end

    return predictor
end

end # module
