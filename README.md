# FNO-based Flow Reconstruction from Lagrangian Vortex Particle Field

# File description

datasets/autodomain_particle_grid_2D.py - helps in defining domains extending over the entire simulation, without manually entering.
datasets/particle_to_grid_2D.py - converts per frame particle data to 2D Eulerian grids

datasets/extract_pair.py - main script for converting h5,xmf to npz

datasets/dataset_gen_2D.py - dataset loader, applies grid embeddings and plot sample input/output
3D version slices 3D data to 2D for positional embedding

datasets/pt_reader.py-  inspect torch data files keys/shapes

data/raw_data - poisson for now, 1,2,... for initial conditions within the simulation for the same geometry
1. airfoil - 1 - 1000 timestep, 400 fdom with 10 step
2. simple wing - 1 - 200 timestep, 150 fdom with 10 step

Data sim files might need corrections for IC (data variability) and sim params (correctness)
Refer readme.md in each data folder for simulation info

data/train - pair_1 corresponds to 1st set of data generation and pair_1_grid for the output of the datasets/particle_to_grid file

rvpm_codes/ - raw codes for simulation (input with resolution) and post-processing (output with a different or same resolution)



# NOTE:
1. pfield and fdom - neuralop
vtk - forces - aeroelastic solver

2. Train-test split can be chronologically ordered if temporal extrapolation needs to be assesed, and it is not a problem if late-time physics is not very different.

3. Completely ignoring 3D data. Find a way!

4. The resolution inputed at Nx,Ny in particle_to_grid_2D is basically the input resolution of the data, while the output resolution of train data is the fdom resolution set during the simulation run. Prediction will be based on this and should still be able to predict at a higher resolution when trained on lower resolution data (fdom output data on lower resolution), get fdom output from model at higher resolution both spatially and temporal prediction.  

5. The model needs input_grid and U_grid on the same fixed (Nx, Ny) grid. That's why particle to grid is run on both input and output (more guarantee to match)

6. Right now the input pair in train and test are both same resolution 64x64x64 for the FNO model due to projection using Gaussian kernels. Try out a lower input pair resolution for training alone, for the evaluation of zero shot super resolution by maintaining the same input pair resolution for test.

7. Train/val/test distribution shift
If the early frames have very different dynamics (startup transient), and the model mostly saw later frames (or random split picked few early ones), it will fail badly there.

Normalization mismatch
If normalization stats are computed on a subset that under‑represents early frames, those samples can be scaled poorly → huge errors.

Model underfit + low capacity
With a small model (low hidden_channels, few epochs), it often only learns “average” behavior — early transients are the hardest.

Data mismatch in resolution
If the test input is 64³ but output truth is 16³, or vice‑versa, you’re effectively comparing different grids — early frames amplify that mismatch.

8. I have a concern in terms of the normalization. Is the feature scaling performed for individual datasets or collectively among the entire dataset, as differences in the datasets completely (in terms of say geometry) might cause different range of values for certain parameters.

9. The mean errors associated with task1 for train test validate for both u and gradu are very different magnitudes.

10. The geometry encoding and conditioning channels (metadata) were not proper before - trying to fix it.

# Question:
1. Are we gonna run simulations and validate for only static cases as in a aerofoil kept at an AOA or even for rotor, flapping cases? (Generalization to unseen geometries and kinematics)

2. Why do we do this grid embedding and positional embedding? (Has to be scraped - no more projections)
Sol: Your raw VPM state is particles at irregular positions. FNO training expects tensors on a fixed regular grid (C, Nx, Ny). So this step deposits particle quantities (like gamma_mag, sigma, vol) onto grid cells and also grids target velocity.Hence, it converts unstructured particle data into a format neural operators can learn from efficiently and consistently.

A regular grid index alone does not explicitly tell the model physical coordinates. Adding coordinate channels (x, y) gives absolute location information, which helps with boundary/location-dependent behavior. In your current repo, this is shown in dataset_gen_2D.py mostly for inspection/demo; the main training file neuraloperator_train.py does not currently add those channels explicitly.


3. Explain in paper clearly why neuraloperators is the thing here - infinite dimensional, operator learning function spaces, super-resolution, not data hungry (zero-shot), etc...

4. Explain in paper clearly how neuraloperators learn and what are they learning?

