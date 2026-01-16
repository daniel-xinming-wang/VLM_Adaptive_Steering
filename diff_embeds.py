# %%
import os
import os.path as op
from typing import List

import numpy as np

# %%
EMBEDS_A_DIR = "Data/Representation/MathV/Qwen3-VL-4B-Thinking-no-image"
EMBEDS_B_DIR = "Data/Representation/MathV/Qwen3-VL-4B-Thinking"
NUM_LAYERS = 36
DIRECTION = "a_minus_b"  # "a_minus_b" or "b_minus_a"

# %%
OUT_DIR = "/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Next/Multi-Modal/Data/Representation/Diff_image_vs_no_image"



def load_layer_paths(root: str, num_layers: int) -> List[str]:
    paths: List[str] = []
    for idx in range(num_layers):
        path = op.join(root, f"embeds_{idx}.npy")
        if not op.exists(path):
            raise FileNotFoundError(f"Missing layer file: {path}")
        paths.append(path)
    return paths



# %%

def main() -> None:
    if DIRECTION not in {"a_minus_b", "b_minus_a"}:
        raise ValueError(f"Unsupported direction: {DIRECTION}")

    os.makedirs(OUT_DIR, exist_ok=True)

    paths_a = load_layer_paths(EMBEDS_A_DIR, NUM_LAYERS)
    paths_b = load_layer_paths(EMBEDS_B_DIR, NUM_LAYERS)

    for idx, (path_a, path_b) in enumerate(zip(paths_a, paths_b)):
        arr_a = np.load(path_a)
        arr_b = np.load(path_b)
        if arr_a.shape != arr_b.shape:
            raise ValueError(
                f"Shape mismatch at layer {idx}: {arr_a.shape} vs {arr_b.shape}"
            )
        if DIRECTION == "a_minus_b":
            diff = arr_a - arr_b
        else:
            diff = arr_b - arr_a
        out_path = op.join(OUT_DIR, f"embeds_{idx}.npy")
        np.save(out_path, diff)

    print(f"[info] saved diffs to: {OUT_DIR}")


if __name__ == "__main__":
    main()


# %%
for i in range(NUM_LAYERS):
    path = os.path.join(OUT_DIR, f"embeds_{i}.npy")
    diff = np.load(path)
    per_sample = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    mean_norm = per_sample.mean()
    std_norm = per_sample.std()
    frob_norm = np.linalg.norm(diff)
    print(f"layer {i:02d} | mean={mean_norm:.4f} std={std_norm:.4f} frob={frob_norm:.4f}")

# %%
def cosine_to_mean(mat: np.ndarray):
    # mat: [N, H] or [N, ...]
    mat2d = mat.reshape(mat.shape[0], -1)
    mean_vec = mat2d.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)
    if mean_norm == 0:
        return None, None
    mean_unit = mean_vec / mean_norm
    vec_norms = np.linalg.norm(mat2d, axis=1)
    valid = vec_norms > 0
    cos = (mat2d[valid] @ mean_unit) / vec_norms[valid]
    return cos, mean_norm

for i in range(NUM_LAYERS):
    path = os.path.join(OUT_DIR, f"embeds_{i}.npy")
    diff = np.load(path)
    cos, mean_norm = cosine_to_mean(diff)
    if cos is None:
        print(f"layer {i:02d} | mean_vec_norm=0")
        continue
    print(
        f"layer {i:02d} | mean_vec_norm={mean_norm:.4f} "
        f"| cos_mean={cos.mean():.4f} std={cos.std():.4f} "
        f"| same_dir_ratio={(cos>0).mean():.3f}"
    )

# %%
mat2d = diff.reshape(diff.shape[0], -1)
mean_vec = mat2d.mean(axis=0)
mean_unit = mean_vec / np.linalg.norm(mean_vec)
cos = (mat2d @ mean_unit) / np.linalg.norm(mat2d, axis=1)

print("min_cos", cos.min(), "p01", np.quantile(cos, 0.01))
print("mean_norm/avg_norm", np.linalg.norm(mean_vec) / np.mean(np.linalg.norm(mat2d, axis=1)))


# %%
neg = (cos <= 0).sum()
total = cos.size
ratio = 1 - neg / total
print("neg", int(neg), "total", int(total), "ratio", ratio)
print("ratio_6dp", f"{ratio:.6f}", "min", cos.min(), "p01", np.quantile(cos, 0.01))


# %%
out_path = "Data/Representation/steering_vectors_image_vs_no_image.npy"
direction = "a_minus_b"
# direction = "b_minus_a"
normalize = False

layer_vecs = []
for i in range(1, NUM_LAYERS + 1):
    a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{i}.npy"))
    b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{i}.npy"))
    diff = a - b if direction == "a_minus_b" else b - a
    vec = diff.mean(axis=0)  # [H]
    if normalize:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    layer_vecs.append(vec)

steering = np.stack(layer_vecs, axis=0)  # [L, H]
np.save(out_path, steering)
print("saved:", out_path, steering.shape)

# %%
import matplotlib.pyplot as plt

K_MAX = 40

def load_layer(path):
    return np.load(path)  # shape [N, H]

def pca_cumvar(X, k_max):
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD: Xc = U S Vt, singular values S
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eigvals = (S ** 2) / (Xc.shape[0] - 1)
    total = eigvals.sum()
    cumvar = np.cumsum(eigvals) / (total + 1e-12)
    return cumvar[:k_max]

plt.figure(figsize=(6, 4))
for layer in range(1, NUM_LAYERS, 5):
    a = load_layer(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
    b = load_layer(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))
    X = np.concatenate([a, b], axis=0)  # [N_total, H]
    cumvar = pca_cumvar(X, K_MAX)
    plt.plot(range(1, len(cumvar)+1), cumvar, label=f"L{layer}")

plt.xlabel("Number of PCs (k)")
plt.ylabel("Cumulative Variance Ratio")
plt.title("PCA Cumulative Variance by Layer")
plt.legend(ncol=4, fontsize=6)
plt.tight_layout()
plt.show()

# %%
STEER_PATH = "Data/Representation/steering_vectors_image_vs_no_image.npy"
NUM_LAYERS = 36
K = 10  # top-k PCs

def pca_project_ratio(X, r, k):
    # X: [N,H], r: [H]
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Ueff = Vt[:k].T  # [H,k]
    r_proj = Ueff @ (Ueff.T @ r)
    return np.linalg.norm(r_proj) / (np.linalg.norm(r) + 1e-12)

steer = np.load(STEER_PATH)  # [L,H]
ratios = []

for layer in range(NUM_LAYERS):
    a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
    b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))
    X = np.concatenate([a, b], axis=0)
    r = steer[layer]
    ratio = pca_project_ratio(X, r, K)
    ratios.append(ratio)
    print(f"layer {layer:02d} | ratio={ratio:.4f}")

plt.figure(figsize=(5,3))
plt.plot(range(NUM_LAYERS), ratios, marker="o")
plt.xlabel("Layer")
plt.ylabel("||P_M r|| / ||r||")
plt.title(f"Projection Energy Ratio (k={K})")
plt.tight_layout()
plt.show()

# %%
scores = []
for layer in range(NUM_LAYERS):
    a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
    b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)
    r = mu_a - mu_b
    r_norm = np.linalg.norm(r) + 1e-12
    r_unit = r / r_norm

    proj_a = a @ r_unit
    proj_b = b @ r_unit
    score = (proj_a.mean() - proj_b.mean()) / (proj_a.std() + proj_b.std() + 1e-12)

    scores.append((layer, score, r_norm))

candidates = [x for x in scores if x[0] >= NUM_LAYERS * 2 // 3]
candidates.sort(key=lambda x: x[1], reverse=True)

print("Top candidates (layer, score, r_norm):")
for item in candidates[:5]:
    print(item)

# %%
def compute_pca_basis(X, k=None, var_thresh=None):
    # X: [N, H]
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    eig = (S ** 2) / (Xc.shape[0] - 1)
    if var_thresh is not None:
        cum = np.cumsum(eig) / (eig.sum() + 1e-12)
        k = int(np.searchsorted(cum, var_thresh) + 1)
    if k is None:
        raise ValueError("Provide k or var_thresh.")
    Ueff = Vt[:k].T  # [H, k]
    return Ueff

def purify_steering(a, b, k=10, var_thresh=None, eps=1e-12):
    # a, b: [N, H]
    r = a.mean(axis=0) - b.mean(axis=0)  # r(l*)
    Ueff = compute_pca_basis(np.concatenate([a, b], axis=0), k=k, var_thresh=var_thresh)
    r_proj = Ueff @ (Ueff.T @ r)
    norm = np.linalg.norm(r_proj)
    if norm > eps:
        r_proj = r_proj / norm
    return r_proj

layer = 24
a = np.load(os.path.join(EMBEDS_A_DIR, f"embeds_{layer}.npy"))
b = np.load(os.path.join(EMBEDS_B_DIR, f"embeds_{layer}.npy"))

r_purified = purify_steering(a, b, k=10)  # or var_thresh=0.8
np.save("Data/Representation/steering_vector_layer24_purified.npy", r_purified)
print("saved", r_purified.shape)

# %%
import os
import re
import numpy as np

root = "/Users/xinmingwang/Library/CloudStorage/OneDrive-NationalUniversityofSingapore/Next/Multi-Modal/Data/Representation/MathV/Qwen3-VL-4B-Thinking"

def layer_idx(path):
    m = re.search(r"embeds_(\d+)\.npy$", path)
    return int(m.group(1)) if m else -1

paths = [
    os.path.join(root, f) for f in os.listdir(root)
    if f.startswith("embeds_") and f.endswith(".npy")
]
paths = sorted(paths, key=layer_idx)

means = []
for p in paths:
    arr = np.load(p, mmap_mode="r")  # [N, H]
    means.append(arr.mean(axis=0).astype(np.float32))

calib = np.stack(means, axis=0)  # [L, H]
out_path = os.path.join(root, "calibration_vectors.npy")
np.save(out_path, calib)
print("saved:", out_path, calib.shape)



