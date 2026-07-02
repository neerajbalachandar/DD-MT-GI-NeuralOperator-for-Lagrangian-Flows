module Task1PythonPredictor

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
    cmd = `$(python_bin) $(helper_py) --server --model $(model_py_path) --meta $(model_meta_npz) --device $(device)`
    proc = open(cmd, "r+")
    closed = Ref(false)

    function predictor(Xfeatures::AbstractMatrix)
        closed[] && error("Task1 Python predictor subprocess is already closed")
        X = Array{Float32}(Xfeatures)
        nrows, ncols = size(X)
        row_major = vec(permutedims(X))
        try
            write(proc, Int64(nrows))
            write(proc, Int64(ncols))
            write(proc, row_major)
            flush(proc)

            outrows = read(proc, Int64)
            outcols = read(proc, Int64)
            if outrows < 0 || outcols < 0
                error("Task1 Python predictor returned invalid shape ($(outrows), $(outcols))")
            end
            buf = Vector{Float32}(undef, Int(outrows * outcols))
            read!(proc, buf)
            return Array{Float32}(permutedims(reshape(buf, Int(outcols), Int(outrows))))
        catch err
            error("Task1 persistent Python predictor failed. Command: $(cmd). Error: $(err)")
        end
    end

    function close_predictor()
        if !closed[]
            try
                write(proc, Int64(-1))
                write(proc, Int64(0))
                flush(proc)
            catch
            end
            close(proc)
            closed[] = true
        end
    end

    return predictor, close_predictor
end

end # module
