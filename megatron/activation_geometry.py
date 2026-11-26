# Copyright (c) 2025, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Geometry of activations (hidden states) during training (activation_geometry_interval,
activation_probe_interval).

Forward hooks sample the same activation_geometry_tokens token positions from every capture point:

    embedding          the embedding output (input to block 00)
    block NN/residual  the residual stream after block NN
    block NN/attention the attention branch output of block NN (attention(ln(x)) + bias)
    block NN/mlp       the MLP branch output of block NN (mlp(ln(x)) + bias)
    final_norm         the final layer norm's output (the input to the unembedding)

so that, with hidden_dropout = 0, block NN/residual = previous residual + block NN/attention +
block NN/mlp for every sampled token, in both parallel (gpt_j_residual) and sequential blocks.

Each captured (n_tokens, hidden) matrix gets the metrics of saturation's
src/metrics/geometric_metrics.py (compute_mtrx_geometric_metrics), computed on the GPU in float64 with
the same definitions, plus mean_token_norm; see matrix_metrics. For each block, branch_metrics compares
the attention and MLP branch outputs of the same tokens.

Capture is on only while train_step runs forward_step for the first micro-batch of a tracked
iteration, and while the probe batch runs; never during the recomputation of checkpointed layers in
backward_step or during evaluation. Assumes model_parallel_size == 1, no pipeline or sequence
parallelism, no MoE and no TransformerEngine layers (checked in ActivationCapture).
"""

import torch
import torch.nn.functional as F

# The metrics of geometric_metrics.compute_mtrx_geometric_metrics that are logged, in order
# (classical_rank is dropped: always full for these matrices; sing_sum and sing_mean are nuc_norm and
# nuc_norm / hidden).
MATRIX_METRICS = (
    "numerical_rank",
    "effective_rank",
    "rankme",
    "alpha",
    "spectral_norm",
    "frobenius_norm",
    "nuc_norm",
    "l1_norm",
    "sing_var",
    "sing_vmr",
    "sing1_sing2",
    "condition_number",
    "determinant",
    "t_sim",
    "t_cos",
    "iso_score",
    "mean_token_norm",
)
BRANCH_METRICS = ("cos", "norm_ratio", "subspace_overlap")


# Eigendecompositions of the (d, d) Gram and covariance matrices run on the CPU up to this d (much
# faster there than float64 on the GPU at small d), on the GPU above it.
CPU_EIGEN_MAX_DIM = 512


def summarize(x):
    """
    GPU stage of matrix_metrics: reduce the (n, d) sample to what the metrics need, in float64 on x's
    device: the Gram matrix x^T x (its eigenvalues are the squared singular values of x), the unbiased
    covariance of the centered tokens, and the scalars l1_norm, mean_token_norm, t_cos and t_sim.
    """
    x = x.to(torch.float64)
    n = x.shape[0]
    mean = x.mean(dim=0)
    centered = x - mean
    gram = x.T @ x
    unit = F.normalize(x, dim=1)
    col_sum = unit.sum(dim=0)
    t_cos = (col_sum @ col_sum - (unit * unit).sum()) / (n * (n - 1))
    t_sim = n * (mean @ mean) / torch.trace(gram)
    scalars = torch.stack([x.abs().sum(), x.norm(dim=1).mean(), t_cos, t_sim])
    return {"n": n, "gram": gram, "cov": centered.T @ centered / (n - 1), "scalars": scalars}


def _eigen_device(summary):
    d = summary["gram"].shape[0]
    return torch.device("cpu") if d <= CPU_EIGEN_MAX_DIM else summary["gram"].device


def metrics_from_summary(summary):
    """
    Geometric metrics from summarize()'s output, as a float64 CPU tensor ordered as MATRIX_METRICS.
    Same definitions as saturation's src/metrics/geometric_metrics.py (compute_mtrx_geometric_metrics):
    - numerical_rank: # singular values >= 0.5 * the largest (numerical_rank(epsilon=0.5));
      effective_rank: exp(entropy of the normalized singular values) (Roy & Vetterli);
      condition_number: largest / smallest singular value > 1e-10; determinant: sum of log singular
      values; sing_var: unbiased variance of the singular values; sing_vmr = sing_var / mean;
      sing1_sing2: first - second singular value; l1_norm: entrywise sum |x_ij|.
    - rankme, alpha: from the PCA eigenspectrum (explained-variance ratios of the centered tokens), as
      get_eigenspectrum / rankme / stringer_get_powerlaw (fit range min(10, d // 10) .. min(100, d)).
    - t_sim: ||mean token||^2 * n / ||x||_F^2; t_cos: mean cosine over ordered pairs i != j.
    - iso_score: IsoScore (Rudman et al.) of the token cloud, from the (unbiased, centered) covariance.
    - mean_token_norm: mean L2 norm of the tokens (unlike the matrix norms, independent of n).
    Singular values come from the eigenvalues of x^T x in float64, accurate unless the condition number
    exceeds ~1e7 (then the smallest ones, so condition_number and determinant, are inaccurate).
    The matrix norms (spectral, frobenius, nuc, l1, determinant, sing_*) grow with n, so they are
    comparable across steps only for a fixed activation_geometry_tokens.
    """
    device = _eigen_device(summary)
    gram, cov = summary["gram"].to(device), summary["cov"].to(device)
    l1_norm, mean_token_norm, t_cos, t_sim = summary["scalars"].to(device)
    d = gram.shape[0]
    s = torch.linalg.eigvalsh(gram).flip(0).clamp(min=0).sqrt()  # singular values, descending
    s_mean = s.mean()
    s_var = s.var()
    nonzero = s[s > 1e-10]
    p = s / (s.sum() + 1e-10)
    p = p[p > 0]

    # PCA eigenspectrum: eigenvalues of the covariance, as explained-variance ratios.
    cov_eigen = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0)
    eigen = cov_eigen / cov_eigen.sum()
    scores = eigen / eigen.abs().sum() + 1e-7
    rankme = torch.exp(-(scores * torch.log(scores)).sum())
    start, end = min(10, d // 10), min(100, d)
    t = torch.arange(start, end, dtype=torch.float64, device=device) + 1
    y = torch.log(eigen.abs() + 1e-12)[start:end]
    design = torch.stack([-torch.log(t), torch.ones_like(t)], dim=1)
    w = (1.0 / t)[:, None]
    coef = torch.linalg.solve(design.T @ (design * w), (w * design).T @ y[:, None]).flatten()

    # IsoScore, as IsoScore.IsoScore: singular values of the covariance (= its eigenvalues, as it is PSD).
    dim = torch.tensor(float(d), dtype=torch.float64, device=device)
    pcs_norm = cov_eigen * torch.sqrt(dim) / torch.linalg.vector_norm(cov_eigen)
    iso_defect = torch.linalg.vector_norm(pcs_norm - 1) / torch.sqrt(2 * (dim - torch.sqrt(dim)))
    iso_score = ((dim - iso_defect**2 * (dim - torch.sqrt(dim))) ** 2 - dim) / (dim * (dim - 1))

    zero = torch.zeros((), dtype=torch.float64, device=device)
    values = {
        "numerical_rank": (s >= 0.5 * s[0]).sum().to(torch.float64),
        "effective_rank": torch.exp(-(p * torch.log(p)).sum()),
        "rankme": rankme,
        "alpha": coef[0],
        "spectral_norm": s[0],
        "frobenius_norm": torch.trace(gram).sqrt(),
        "nuc_norm": s.sum(),
        "l1_norm": l1_norm,
        "sing_var": s_var,
        "sing_vmr": s_var / s_mean,
        "sing1_sing2": s[0] - s[1],
        "condition_number": s[0] / nonzero.min() if nonzero.numel() else zero,
        "determinant": torch.log(s).sum(),
        "t_sim": t_sim,
        "t_cos": t_cos,
        "iso_score": iso_score,
        "mean_token_norm": mean_token_norm,
    }
    return torch.stack([values[name] for name in MATRIX_METRICS]).cpu()


def matrix_metrics(x):
    """Geometric metrics of x (n, d), n tokens of d dims (see metrics_from_summary)."""
    return metrics_from_summary(summarize(x))


def summarize_branches(attention, mlp):
    """GPU stage of branch_metrics: mean token cosine and mean token norms of the two branches."""
    a, m = attention.to(torch.float64), mlp.to(torch.float64)
    return torch.stack([F.cosine_similarity(a, m, dim=1).mean(), a.norm(dim=1).mean(), m.norm(dim=1).mean()])


def branch_metrics_from_summaries(branch_summary, attention_summary, mlp_summary):
    """
    Redundancy of a block's attention and MLP branch outputs (n, d) for the same tokens, as a float64 CPU
    tensor ordered as BRANCH_METRICS:
    - cos: mean over tokens of cos(attention_i, mlp_i);
    - norm_ratio: mean ||attention_i|| / mean ||mlp_i||;
    - subspace_overlap: ||Va^T Vm||_F^2 / k for the top-k right singular vectors Va, Vm of the two
      (uncentered) matrices (top-k eigenvectors of their Gram matrices), k = max(1, d // 8):
      1 = the same k-dim subspace, k / d expected for random subspaces.
    """
    device = _eigen_device(attention_summary)
    cos, a_norm, m_norm = branch_summary.to(device)
    k = max(1, attention_summary["gram"].shape[0] // 8)
    va = torch.linalg.eigh(attention_summary["gram"].to(device)).eigenvectors[:, -k:]
    vm = torch.linalg.eigh(mlp_summary["gram"].to(device)).eigenvectors[:, -k:]
    return torch.stack([cos, a_norm / m_norm, (va.T @ vm).pow(2).sum() / k]).cpu()


def branch_metrics(attention, mlp):
    """branch_metrics_from_summaries for (n, d) attention and MLP branch outputs of the same tokens."""
    return branch_metrics_from_summaries(
        summarize_branches(attention, mlp), summarize(attention), summarize(mlp)
    )


def _finite(x):
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    return not torch.is_tensor(x) or bool(torch.isfinite(x).all())


def _or_nan(n, fn, *args):
    """fn(*args), or n NaNs if an input is non-finite (e.g. after an fp16 overflow) or fn fails (an
    eigendecomposition that doesn't converge), so a diverging run keeps training. Purely local: every
    rank still reaches compute()'s all-reduce."""
    if all(_finite(a) for a in args):
        try:
            return fn(*args)
        except RuntimeError as e:
            print(f"activation geometry: {fn.__name__} failed ({e!r}); logging NaN", flush=True)
    return torch.full((n,), float("nan"), dtype=torch.float64)


class ActivationCapture:
    """
    Forward hooks that sample activation_geometry_tokens token positions from every capture point
    (see the module docstring) while capturing is on.

        capture.start(seed)    # sample positions with torch.Generator().manual_seed(seed)
        ... forward pass ...
        capture.stop()
        metrics = capture.compute()   # {"<label>/<metric>": float}, averaged over data-parallel ranks

    Positions are drawn once the embedding hook sees the batch shape, uniformly without replacement
    over (sequence, position) pairs, from a dedicated generator (the global RNG is untouched).
    """

    def __init__(self, model, neox_args):
        from megatron import mpu
        from megatron.model.transformer import (
            NormPipe,
            ParallelTransformerLayer,
            ParallelTransformerLayerPipe,
        )
        from megatron.model.word_embeddings import EmbeddingPipe

        assert mpu.get_model_parallel_world_size() == 1, "activation geometry assumes model_parallel_size == 1"
        assert not neox_args.is_pipe_parallel, "activation geometry assumes no pipeline parallelism"
        assert not neox_args.sequence_parallel, "activation geometry assumes no sequence parallelism"
        assert neox_args.moe_num_experts == 1, "activation geometry assumes no MoE layers"
        assert not (neox_args.te_mha or neox_args.te_fp8_mha or neox_args.te_layernorm_mlp), (
            "activation geometry assumes no TransformerEngine layers"
        )
        self.n_tokens = neox_args.activation_geometry_tokens
        self.active = False
        self.samples = {}
        self.handles = []

        modules = list(model.module.modules())
        embeddings = [m for m in modules if isinstance(m, EmbeddingPipe)]
        norms = [m for m in modules if isinstance(m, NormPipe)]
        layers = [m for m in modules if isinstance(m, ParallelTransformerLayerPipe)]
        assert len(embeddings) == 1 and len(norms) == 1 and layers, (
            f"activation geometry expects 1 EmbeddingPipe, 1 NormPipe and transformer layers, found "
            f"{len(embeddings)}, {len(norms)} and {len(layers)}"
        )
        assert len(layers) == neox_args.num_layers, (
            "activation geometry expects every layer to be a transformer layer (no gmlp/mamba/rwkv)"
        )
        assert not any(
            isinstance(m, ParallelTransformerLayer) and not isinstance(m, ParallelTransformerLayerPipe)
            for m in modules
        )
        self.layer_numbers = sorted(layer.layer_number for layer in layers)
        self.final_norm = norms[0]

        # Layouts: the embedding and final norm see [b, s, h]; the transformer layers [s, b, h].
        self._hook(embeddings[0], "embedding", "bsh", first=True)
        for layer in layers:
            prefix = f"block {layer.layer_number:02d}"
            self._hook(layer, f"{prefix}/residual", "sbh")
            self._hook(layer.attention, f"{prefix}/attention", "sbh", add_bias=True)
            self._hook(layer.mlp, f"{prefix}/mlp", "sbh", add_bias=True)
        self._hook(norms[0], "final_norm", "bsh")

    def _hook(self, module, label, layout, first=False, add_bias=False):
        def hook(_module, _inputs, output):
            if not self.active:
                return
            if add_bias:  # (output, bias); output is [output, present] when caching keys/values
                output, bias = output
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if bias is not None:
                    output = output + bias
            elif isinstance(output, tuple):  # (hidden_states, attention_mask)
                output = output[0]
            if first:
                self._draw_positions(output.shape[0], output.shape[1])
            b, s = (output.shape[0], output.shape[1]) if layout == "bsh" else (output.shape[1], output.shape[0])
            if (b, s) != self.batch_shape:
                raise RuntimeError(
                    f"activation geometry: {label} has {layout} shape {tuple(output.shape)}, expected "
                    f"(batch, seq) = {self.batch_shape}"
                )
            with torch.no_grad():
                if layout == "bsh":
                    sample = output.detach()[self.batch_index, self.seq_index]
                else:
                    sample = output.detach()[self.seq_index, self.batch_index]
                self.samples[label] = sample.float()

        self.handles.append(module.register_forward_hook(hook))

    def _draw_positions(self, batch, seq):
        generator = torch.Generator().manual_seed(self.seed)
        n = min(self.n_tokens, batch * seq)
        flat = torch.randperm(batch * seq, generator=generator)[:n]
        device = torch.cuda.current_device()
        self.batch_shape = (batch, seq)
        self.batch_index = (flat // seq).to(device)
        self.seq_index = (flat % seq).to(device)

    def start(self, seed):
        self.seed = seed
        self.samples = {}
        self.batch_shape = None
        self.active = True

    def stop(self):
        self.active = False

    def labels(self):
        labels = ["embedding"]
        for number in self.layer_numbers:
            labels += [f"block {number:02d}/{name}" for name in ("residual", "attention", "mlp")]
        return labels + ["final_norm"]

    def compute(self):
        """{"<label>/<metric>": float} of the last capture, averaged over data-parallel ranks; clears it."""
        missing = [label for label in self.labels() if label not in self.samples]
        if missing:
            raise RuntimeError(f"activation geometry: nothing captured for {missing}")
        with torch.no_grad():
            summaries = {label: summarize(self.samples[label]) for label in self.labels()}
            branches = {
                number: summarize_branches(
                    self.samples[f"block {number:02d}/attention"], self.samples[f"block {number:02d}/mlp"]
                )
                for number in self.layer_numbers
            }
            keys, values = [], []
            for label in self.labels():
                keys += [f"{label}/{name}" for name in MATRIX_METRICS]
                values.append(_or_nan(len(MATRIX_METRICS), metrics_from_summary, summaries[label]))
            for number in self.layer_numbers:
                prefix = f"block {number:02d}"
                keys += [f"{prefix}/branch/{name}" for name in BRANCH_METRICS]
                values.append(_or_nan(
                    len(BRANCH_METRICS), branch_metrics_from_summaries,
                    branches[number], summaries[f"{prefix}/attention"], summaries[f"{prefix}/mlp"],
                ))
            values = torch.cat(values)
            if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
                values = values.to(torch.cuda.current_device())
                torch.distributed.all_reduce(values)
                values /= torch.distributed.get_world_size()
        self.samples = {}
        return dict(zip(keys, values.tolist()))

    def residual_check(self):
        """
        Max relative errors of the identities that hold if every hook captured the right tensor and
        tokens (hidden_dropout = 0): block residual = previous residual + attention + mlp, and
        final_norm = final layer norm(last residual). Errors at bf16 rounding level (~1e-2) are fine.
        Call before compute(), which clears the samples.
        """
        rel = lambda a, b: ((a - b).norm(dim=1) / b.norm(dim=1).clamp(min=1e-12)).max().item()
        errors, previous = {}, self.samples["embedding"]
        for number in self.layer_numbers:
            prefix = f"block {number:02d}"
            residual = self.samples[f"{prefix}/residual"]
            errors[prefix] = rel(previous + self.samples[f"{prefix}/attention"] + self.samples[f"{prefix}/mlp"], residual)
            previous = residual
        norm = self.final_norm.norm
        dtype = next(norm.parameters()).dtype if any(True for _ in norm.parameters()) else previous.dtype
        with torch.no_grad():
            # The norm ran on the bf16/fp16 residual stream; the samples are float copies of it.
            expected = norm(previous.to(dtype)).float()
        errors["final_norm"] = rel(expected, self.samples["final_norm"])
        return errors
