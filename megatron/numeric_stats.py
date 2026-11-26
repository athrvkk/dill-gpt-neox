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
fp16-range and output-logit statistics during training (numeric_stats_interval).

Forward hooks see the full tensors of the first training micro-batch of every tracked iteration at

    embedding                       the embedding output (input to block 00)
    block NN/input_layernorm        the attention LayerNorm's output
    block NN/post_attention_layernorm  the MLP LayerNorm's output
    block NN/attention              the attention branch output (+ bias)
    block NN/mlp                    the MLP branch output (+ bias)
    block NN/residual               the residual stream after block NN
    final_norm                      the final LayerNorm's output
    logits                          the unembedding output

and record, as "numerics/<point>/<stat>":

    absmax            max |x| (nan/inf if any entry is)
    frac above 2^15   fraction of entries with |x| > 2^15, within 2x of the fp16 max 65504
    frac subnormal    fraction of entries with 0 < |x| < 2^-14, where fp16 loses precision
    frac nonfinite    fraction of inf / nan entries

For the logits, absmax is logged as "output logit max abs", plus "output logit std" (std over the
vocab, mean over tokens) and "output logit norm" (L2 norm over the vocab, mean over tokens), as in
Chen et al. 2026 (arXiv:2610.38277). From the weights: "output embedding mean norm", the norm of the
mean row of the unembedding matrix (its drift under Adam shifts every logit, Chen et al. 2026), and
"numerics/<norm>/gain absmax" / "bias absmax" for every LayerNorm.

Capture is on only while train_step runs forward_step for the first micro-batch of a tracked
iteration, so never during the recomputation of checkpointed layers or during evaluation. Assumes
model_parallel_size == 1 and no pipeline parallelism (checked in NumericStats).
"""

import torch

FP16_NEAR_MAX = 2.0**15  # fp16 max is 65504
FP16_MIN_NORMAL = 2.0**-14
# Elements per fp32 chunk of the logits (256 MB), so the full [b, s, vocab] tensor is never copied.
LOGIT_CHUNK = 1 << 26


def _first_tensor(output, add_bias=False):
    """The activation in a module output: (out, bias) for attention / mlp, (hidden, mask) for layers."""
    if add_bias:  # (output, bias); output is [output, present] when caching keys/values
        output, bias = output
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output if bias is None else output + bias
    return output[0] if isinstance(output, (list, tuple)) else output


@torch.no_grad()
def range_counts(x):
    """[absmax, # above 2^15, # subnormal, # nonfinite, # entries] of x, float64 on x's device."""
    ax = x.abs()
    return torch.stack(
        [
            ax.amax().double(),
            (ax > FP16_NEAR_MAX).sum().double(),
            ((ax < FP16_MIN_NORMAL) & (ax > 0)).sum().double(),
            (~torch.isfinite(x)).sum().double(),
            torch.tensor(float(x.numel()), dtype=torch.float64, device=x.device),
        ]
    )


@torch.no_grad()
def logit_counts(logits):
    """range_counts of the logits plus [sum over tokens of std, sum of L2 norm, # tokens], in chunks."""
    vocab = logits.shape[-1]
    rows = logits.reshape(-1, vocab)
    step = max(1, LOGIT_CHUNK // vocab)
    total = None
    for i in range(0, rows.shape[0], step):
        chunk = rows[i : i + step].float()
        var, mean = torch.var_mean(chunk, dim=-1, unbiased=False)
        norm = torch.sqrt((var + mean**2) * vocab)  # ||x||^2 = vocab * (var + mean^2)
        counts = torch.cat(
            [
                range_counts(chunk),
                torch.stack([var.sqrt().sum(), norm.sum()]).double(),
                torch.tensor([float(chunk.shape[0])], dtype=torch.float64, device=chunk.device),
            ]
        )
        total = counts if total is None else torch.cat(
            [torch.maximum(total[:1], counts[:1]), total[1:] + counts[1:]]
        )
    return total


class NumericStats:
    """
    Forward hooks recording numerics of the capture points (see the module docstring) while on.

        stats.start()
        ... forward pass ...
        stats.stop()
        metrics = stats.compute()   # {name: 0-dim tensor}, reduced over data-parallel ranks
    """

    def __init__(self, model, neox_args):
        from megatron import mpu
        from megatron.model.transformer import NormPipe, ParallelLinearPipe, ParallelTransformerLayerPipe
        from megatron.model.word_embeddings import EmbeddingPipe

        assert mpu.get_model_parallel_world_size() == 1, "numeric stats assume model_parallel_size == 1"
        assert not neox_args.is_pipe_parallel, "numeric stats assume no pipeline parallelism"
        self.active = False
        self.counts = {}
        modules = list(model.module.modules())
        embeddings = [m for m in modules if isinstance(m, EmbeddingPipe)]
        norms = [m for m in modules if isinstance(m, NormPipe)]
        unembeddings = [m for m in modules if isinstance(m, ParallelLinearPipe)]
        layers = [m for m in modules if isinstance(m, ParallelTransformerLayerPipe)]
        assert len(embeddings) == len(norms) == len(unembeddings) == 1 and layers, (
            f"numeric stats expect 1 EmbeddingPipe, NormPipe and ParallelLinearPipe and transformer "
            f"layers, found {len(embeddings)}, {len(norms)}, {len(unembeddings)} and {len(layers)}"
        )
        self.unembedding = unembeddings[0].final_linear.weight
        self.layer_norms = {"final_norm": norms[0].norm}

        self._hook(embeddings[0], "embedding")
        for layer in layers:
            prefix = f"block {layer.layer_number:02d}"
            for name in ("input_layernorm", "post_attention_layernorm"):
                if hasattr(layer, name):
                    self._hook(getattr(layer, name), f"{prefix}/{name}")
                    self.layer_norms[f"{prefix}/{name}"] = getattr(layer, name)
            self._hook(layer.attention, f"{prefix}/attention", add_bias=True)
            self._hook(layer.mlp, f"{prefix}/mlp", add_bias=True)
            self._hook(layer, f"{prefix}/residual")
        self._hook(norms[0], "final_norm")
        self._hook(unembeddings[0], "logits", logits=True)

    def _hook(self, module, label, add_bias=False, logits=False):
        def hook(_module, _inputs, output):
            if self.active:
                x = _first_tensor(output, add_bias).detach()
                self.counts[label] = logit_counts(x) if logits else range_counts(x)

        module.register_forward_hook(hook)

    def start(self):
        self.counts = {}
        self.active = True

    def stop(self):
        self.active = False

    @torch.no_grad()
    def compute(self):
        """{name: 0-dim float32 tensor} of the last capture (see the module docstring); clears it."""
        if "logits" not in self.counts:
            raise RuntimeError("numeric stats: nothing captured")
        labels = list(self.counts)
        counts = torch.stack([self.counts[label][:5] for label in labels])  # [points, 5]
        logit_extra = self.counts["logits"][5:].clone()  # [std sum, norm sum, tokens]
        absmax = counts[:, 0].contiguous()
        sums = torch.cat([counts[:, 1:].reshape(-1), logit_extra])
        torch.distributed.all_reduce(absmax, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(sums)
        sums, (std_sum, norm_sum, tokens) = sums[:-3].view(-1, 4), sums[-3:]
        stats = {}
        for label, a, (above, subnormal, nonfinite, n) in zip(labels, absmax, sums):
            key = "output logit max abs" if label == "logits" else f"numerics/{label}/absmax"
            stats[key] = a
            stats[f"numerics/{label}/frac above 2^15"] = above / n
            stats[f"numerics/{label}/frac subnormal"] = subnormal / n
            stats[f"numerics/{label}/frac nonfinite"] = nonfinite / n
        stats["output logit std"] = std_sum / tokens
        stats["output logit norm"] = norm_sum / tokens
        # Weights are identical on all data-parallel ranks.
        stats["output embedding mean norm"] = self.unembedding.float().mean(dim=0).norm()
        for label, norm in self.layer_norms.items():
            gain = getattr(norm, "weight", getattr(norm, "scale", None))  # RMSNorm calls it scale
            for name, param in (("gain", gain), ("bias", getattr(norm, "bias", None))):
                if param is not None:
                    stats[f"numerics/{label}/{name} absmax"] = param.detach().abs().max()
        self.counts = {}
        return {k: v.float() for k, v in stats.items()}
