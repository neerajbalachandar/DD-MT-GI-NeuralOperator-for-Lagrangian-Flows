Run code: ML

TASK1_PYTHON=./Flow-reconstruction-in-VPM-using-FNO/neuralop-env/bin/python \
julia --project=. \
  ./Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/run_task1_flowunsteady.jl \
  --sim-file ./Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/simulation.jl \
  --mode surrogate_ml \
  --device auto \
  --save-dir ./Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/output/runtime_surrogate_ml

Run code + comparison (FMM)

# Create the missing output directory first
mkdir -p Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/output/benchmark_task1

# Run the background pipeline
TASK1_PIPELINE_PYTHON=$(pwd)/Flow-reconstruction-in-VPM-using-FNO/neuralop-env/bin/python \
TASK1_PYTHON=$(pwd)/Flow-reconstruction-in-VPM-using-FNO/neuralop-env/bin/python \
TASK1_DEVICE=auto \
nohup ./Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/run.sh \
  > Flow-reconstruction-in-VPM-using-FNO/FINAL/julia_time_forward/output/benchmark_task1/overnight_driver.log 2>&1 &





run.sh > benchmark_comparison_task1.py > run_task1_flowunsteady.jl > flowunsteady_callback.jl








