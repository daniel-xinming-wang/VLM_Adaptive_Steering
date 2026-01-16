# %%
import numpy as np
import pandas as pd

# %%
response = pd.read_json("Data/Eval/MathV/Qwen3-VL-4B-Thinking/MathV-Qwen3-VL-4B-Thinking_0.0_all_eval_vote_num1.json")

# %%
response["llm_reasoning_token_num"] = pd.to_numeric(response["llm_reasoning_token_num"].str[0])

# %%
response.groupby("level")["llm_reasoning_token_num"].mean()

# %%
response["is_correct"] = response["is_correct"].str[0]

# %%
df = response.copy()

result = (
    df.groupby(["level", "is_correct"])["llm_reasoning_token_num"]
      .mean()
)
print(result)


# %%
stat = (
    response.groupby(["level", "cannot_see"])
      .agg(
          n=("accuracy", "size"),
          acc_mean=("accuracy", "mean"),
          acc_median=("accuracy", "median"),
          len_mean=("llm_reasoning_token_num", "mean"),
          len_median=("llm_reasoning_token_num", "median"),
      )
)
print(stat)


# %%
delta = (
    response.groupby(["level", "cannot_see"])[["accuracy","llm_reasoning_token_num"]]
      .mean()
      .unstack("cannot_see")
)
delta["acc_drop"] = delta["accuracy"][True] - delta["accuracy"][False]
delta["len_increase"] = delta["llm_reasoning_token_num"][True] - delta["llm_reasoning_token_num"][False]
print(delta[["acc_drop","len_increase"]])


# %%
df2 = response.copy()
df2["reasoning"] = df2["llm_reasoning"].str[0]
df2["cannot_see"] = df2["reasoning"].str.contains("can't see", case=False, na=False)

df2[(df2["cannot_see"]) & (df2["is_correct"] == 1)][
    ["question_id", "image", "question", "llm_final_answer", "reasoning"]
].head(5)


# %%
s = response["llm_reasoning"].str[0]

response["cannot_see"] = s.str.contains("can't see", case=False, na=False)


# %%
cannot_see_result = (
    response.groupby(["level", "cannot_see"])["llm_reasoning_token_num"]
      .mean()
)
print(cannot_see_result)

# %%
print(response.groupby("cannot_see")["accuracy"].mean())

# %%
cannot_see_result2 = (
    response.groupby(["level", "cannot_see"])["accuracy"]
      .mean()
)
print(cannot_see_result2)

# %%
response_8b = pd.read_json("Data/Eval/MathV/Qwen3-VL-8B-Thinking/MathV-Qwen3-VL-8B-Thinking_0.0_all_eval_vote_num1.json")

# %%
response_8b["llm_reasoning_token_num"] = pd.to_numeric(response_8b["llm_reasoning_token_num"].str[0])
response_8b["is_correct"] = response_8b["is_correct"].str[0]

# %%
result_8b = (
    response_8b.groupby(["level", "is_correct"])["llm_reasoning_token_num"]
      .mean()
)
print(result_8b)


# %%
s_8b = response_8b["llm_reasoning"].str[0]

response_8b["cannot_see"] = s_8b.str.contains("can't see", case=False, na=False)


# %%
cannot_see_result_8b = (
    response_8b.groupby(["level", "cannot_see"])["llm_reasoning_token_num"]
      .mean()
)
print(cannot_see_result_8b)

# %%
print(response_8b.groupby("cannot_see")["accuracy"].mean())

# %%
cannot_see_result2_8b = (
    response_8b.groupby(["level", "cannot_see"])["accuracy"]
      .mean()
)
print(cannot_see_result2_8b)

# %%
response_8b["lengthsmall"] = response_8b["avg_llm_reasoning_token_num"] < response_8b["avg_llm_reasoning_token_num"].median()

# %%
import inspect
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

representation_path = "Data/Representation/MathV/Qwen3-VL-8B-Thinking"
layer_id = 25
emb_path = os.path.join(representation_path, f"embeds_{layer_id}.npy")

emb = np.load(emb_path)
"""
if "cannot_see" not in response_8b.columns:
    s_8b = response_8b["llm_reasoning"].str[0]
    response_8b["cannot_see"] = s_8b.str.contains("can't see", case=False, na=False)
"""
assert emb.shape[0] == len(response_8b), "embeddings/sample count mismatch"

mask = response_8b["cannot_see"].values
group = np.where(mask, "cannot_see=True", "cannot_see=False")

pca = PCA(n_components=50, random_state=42)
emb_50 = pca.fit_transform(emb)

tsne_kwargs = dict(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init="pca",
    random_state=42,
    metric="cosine",
)
if "n_iter" in inspect.signature(TSNE).parameters:
    tsne_kwargs["n_iter"] = 2000

coords = TSNE(**tsne_kwargs).fit_transform(emb_50)

df_tsne = pd.DataFrame({
    "tsne_x": coords[:, 0],
    "tsne_y": coords[:, 1],
    "group": group,
})

centroids = (
    df_tsne.groupby("group")[["tsne_x", "tsne_y"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 6))
palette = {"cannot_see=False": "#1f77b4", "cannot_see=True": "#d62728"}

for g in ["cannot_see=False", "cannot_see=True"]:
    df_g = df_tsne[df_tsne["group"] == g]
    plt.scatter(df_g["tsne_x"], df_g["tsne_y"],
                s=18, alpha=0.6, label=g, color=palette[g])

# centroid points in t-SNE space
for g, marker in [("cannot_see=False", "X"), ("cannot_see=True", "P")]:
    c = centroids[centroids["group"] == g]
    plt.scatter(c["tsne_x"], c["tsne_y"],
                s=220, marker=marker, label=f"Mean {g.split('=')[-1]}",
                color=palette[g], edgecolor="black", linewidth=1)

plt.legend(markerscale=1.2)
plt.title(f"t-SNE (Layer {layer_id}) - cannot_see True vs False")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()


# %%
import inspect
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

representation_path = "Data/Representation/MathV/Qwen3-VL-8B-Thinking"
layer_id = 27
emb_path = os.path.join(representation_path, f"embeds_{layer_id}.npy")

emb = np.load(emb_path)

"""
if "cannot_see" not in response_8b.columns:
    s_8b = response_8b["llm_reasoning"].str[0]
    response_8b["cannot_see"] = s_8b.str.contains("can't see", case=False, na=False)
"""
    
assert emb.shape[0] == len(response_8b), "embeddings/sample count mismatch"

pca = PCA(n_components=50, random_state=42)
emb_50 = pca.fit_transform(emb)

tsne_kwargs = dict(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init="pca",
    random_state=42,
    metric="cosine",
)
if "n_iter" in inspect.signature(TSNE).parameters:
    tsne_kwargs["n_iter"] = 2000

coords = TSNE(**tsne_kwargs).fit_transform(emb_50)

df_tsne = pd.DataFrame({
    "tsne_x": coords[:, 0],
    "tsne_y": coords[:, 1],
    "cannot_see": response_8b["cannot_see"].values,
})

plt.figure(figsize=(8, 6))
mask = df_tsne["cannot_see"].values
plt.scatter(df_tsne.loc[~mask, "tsne_x"], df_tsne.loc[~mask, "tsne_y"],
            s=18, alpha=0.6, label="cannot_see=False")
plt.scatter(df_tsne.loc[mask, "tsne_x"], df_tsne.loc[mask, "tsne_y"],
            s=18, alpha=0.6, label="cannot_see=True")
plt.legend(markerscale=1.5)
plt.title(f"t-SNE (Layer {layer_id}) - cannot_see True vs False")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()


# %%
import inspect
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

representation_path = "Data/Representation/MathV/Qwen3-VL-8B-Thinking"
layer_id = 27
emb_path = os.path.join(representation_path, f"embeds_{layer_id}.npy")

emb = np.load(emb_path)

lengths = response_8b["avg_llm_reasoning_token_num"].to_numpy()
n = len(lengths)
assert n > 0, "no samples found"

low_thr = np.quantile(lengths, 0.1)
high_thr = np.quantile(lengths, 0.9)
idx_short = np.where(lengths <= low_thr)[0]
idx_long = np.where(lengths >= high_thr)[0]
print(f"short={len(idx_short)}, long={len(idx_long)}")

emb_short = emb[idx_short]
emb_long = emb[idx_long]
X = np.vstack([emb_short, emb_long])
labels = (["short"] * len(emb_short) + ["long"] * len(emb_long))

pca = PCA(n_components=50, random_state=42)
X_50 = pca.fit_transform(X)

tsne_kwargs = dict(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    init="pca",
    random_state=42,
    metric="cosine",
)
if "n_iter" in inspect.signature(TSNE).parameters:
    tsne_kwargs["n_iter"] = 2000

coords = TSNE(**tsne_kwargs).fit_transform(X_50)

df_tsne = pd.DataFrame({
    "tsne_x": coords[:, 0],
    "tsne_y": coords[:, 1],
    "group": labels,
})

centroids = (
    df_tsne.groupby("group")[["tsne_x", "tsne_y"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(8, 6))
palette = {"short": "#1f77b4", "long": "#d62728"}

for g in ["short", "long"]:
    df_g = df_tsne[df_tsne["group"] == g]
    plt.scatter(df_g["tsne_x"], df_g["tsne_y"],
                s=18, alpha=0.6, label=g, color=palette[g])

# centroid points in t-SNE space
for g, marker in [("short", "X"), ("long", "P")]:
    c = centroids[centroids["group"] == g]
    plt.scatter(c["tsne_x"], c["tsne_y"],
                s=220, marker=marker, label=f"Mean {g}",
                color=palette[g], edgecolor="black", linewidth=1)

plt.legend(markerscale=1.2)
plt.title(f"t-SNE (Layer {layer_id}) - shortest 10% vs longest 10%")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.show()


# %%



