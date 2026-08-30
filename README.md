## A Geometry-Informed, Multi-Task Neural Operator for Nonlocal Kernel Interactions in Lagrangian Flows

### The architecture DD-MT-GINO is a multi-task neural operator that utilizes a shared latent space to predict two different outputs having a similar underlying PDE, using a dual decoder. This architecture is based on the GINO (Geometry-Informed Neural Operator) by Z. Li et al.

Arxiv publication:

The main codes are:
1. particle evolution.py: Training for the evolution of vortex particles according to $\textit{rVPM}$, for the reduced order simulation of flow.
2. field_reconstruction.py: Training for the reconstruction of the velocity field from the Lagrangian field.
3. evaluation.ipynb: Testing and evaluation of the trained model, with analysis for shared latent space. 

