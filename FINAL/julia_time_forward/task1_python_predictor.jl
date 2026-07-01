module Task1PythonPredictor

using DelimitedFiles

export build_predictor

"""
Build Python-backed predictor closure:
  Y = predictor(Xfeatures)
where Xfeatures is Float32 matrix (N,Fin), Y is Float32 matrix (N,12).

This intentionally runs Python out-of-process. Embedding conda Python through
PythonCall can abort Julia when binary Python extensions disagree with Julia's
loaded libraries.
"""
function build_predictor(model_py_path::String, model_meta_npz::String; device::String="auto")
    python_bin = get(ENV, "TASK1_PYTHON", "python3")
    helper_dir = @__DIR__
    helper_py = joinpath(helper_dir, "task1_python_predictor.py")
    scratch_dir = mktempdir(; prefix="task1_ugradu_predict_")

    function predictor(Xfeatures::AbstractMatrix)
        input_csv = joinpath(scratch_dir, "features.csv")
        output_csv = joinpath(scratch_dir, "predictions.csv")
        writedlm(input_csv, Array{Float32}(Xfeatures), ',')

        cmd = `$(python_bin) $(helper_py) --predict-csv $(input_csv) --output-csv $(output_csv) --model $(model_py_path) --meta $(model_meta_npz) --device $(device)`
        try
            run(cmd)
        catch err
            error("Task1 Python predictor subprocess failed. Command: $(cmd). Error: $(err)")
        end

        Y = readdlm(output_csv, ',', Float32)
        if ndims(Y) == 1
            Y = reshape(Y, 1, :)
        end
        return Array{Float32}(Y)
    end

    return predictor
end

end # module
