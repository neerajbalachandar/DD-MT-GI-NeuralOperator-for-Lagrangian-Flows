import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Load dataset
# ============================================================

data = np.load(
    "final-2/output/particle_ugradu_dataset.npz",
    allow_pickle=True
)

# ============================================================
# Print dataset structure
# ============================================================

print("\nAVAILABLE KEYS:\n")

for key in data.files:

    obj = data[key]

    print(f"{key}")
    print(f"type   : {type(obj)}")

    if hasattr(obj, "shape"):
        print(f"shape  : {obj.shape}")

    if hasattr(obj, "dtype"):
        print(f"dtype  : {obj.dtype}")

    print("-" * 60)

# ============================================================
# Main tensors
# ============================================================

X = data["inputs_t"]
Y = data["targets_ugradu"]

Xn = data["inputs_t_norm"]
Yn = data["targets_ugradu_norm"]

feature_names = data["feature_names"]
target_names = data["target_names"]

# ============================================================
# Verify normalization
# ============================================================

mu = data["in_mean"][0]
sigma = data["in_std"][0]

x = X[100]
xn = Xn[100]

xr = xn * sigma + mu

print("\nNormalization check:")
print(np.allclose(x, xr))

# ============================================================
# Feature statistics
# ============================================================

print("\nINPUT FEATURE STATISTICS:\n")

for i, name in enumerate(feature_names):

    col = X[:, i]

    print(f"{name}")
    print(f"mean = {col.mean():.6e}")
    print(f"std  = {col.std():.6e}")
    print(f"min  = {col.min():.6e}")
    print(f"max  = {col.max():.6e}")
    print("-" * 40)

# ============================================================
# Near-body particle fraction
# ============================================================

near = X[:, 11]

print("\nFraction of near-body particles:")
print(np.mean(near))

# ============================================================
# Particle-count evolution
# ============================================================

frame_ranges = data["frame_ranges"]

counts = []

for fr in frame_ranges:

    counts.append(fr[4])

counts = np.array(counts)

print("\nParticle count statistics:")
print("min =", counts.min())
print("max =", counts.max())
print("mean =", counts.mean())

plt.figure(figsize=(8,4))

plt.plot(counts)

plt.xlabel("Frame")
plt.ylabel("Particle count")
plt.title("Particle-count evolution")

plt.tight_layout()
plt.show()

# ============================================================
# Temporal evolution of Gamma_x
# ============================================================

means = []

for fr in frame_ranges:

    s = fr[2]
    e = fr[3]

    means.append(X[s:e, 3].mean())

means = np.array(means)

plt.figure(figsize=(8,4))

plt.plot(means)

plt.xlabel("Frame")
plt.ylabel("Mean Gamma_x")
plt.title("Temporal evolution of mean circulation")

plt.tight_layout()
plt.show()

# ============================================================
# Spatial particle visualization
# ============================================================

idx = np.random.choice(len(X), 100000, replace=False)

pts = X[idx, :3]

plt.figure(figsize=(6,6))

plt.scatter(
    pts[:,0],
    pts[:,1],
    s=0.1
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Particle spatial distribution")

plt.axis("equal")

plt.tight_layout()
plt.show()

# ============================================================
# Target magnitude analysis
# ============================================================

vel_mag = np.linalg.norm(Y[:, :3], axis=1)

grad_mag = np.linalg.norm(Y[:, 3:], axis=1)

print("\nVelocity magnitude statistics:")
print("mean =", vel_mag.mean())
print("max  =", vel_mag.max())

print("\nGradient magnitude statistics:")
print("mean =", grad_mag.mean())
print("max  =", grad_mag.max())

# ============================================================
# Correlation matrix
# ============================================================

sample_idx = np.random.choice(len(X), 100000, replace=False)

corr = np.corrcoef(X[sample_idx].T)

plt.figure(figsize=(8,8))

plt.imshow(corr)

plt.colorbar()

plt.xticks(
    range(len(feature_names)),
    feature_names,
    rotation=90
)

plt.yticks(
    range(len(feature_names)),
    feature_names
)

plt.title("Input feature correlation matrix")

plt.tight_layout()
plt.show()