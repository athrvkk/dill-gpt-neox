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
Redundancy of each block's attention and MLP branches (branch_grad_interval, branch_probe_interval).

BranchGradSimilarity (training micro-batch): forward pre-hooks put an identity node (x.view_as(x)) in
front of each block's input_layernorm and post_attention_layernorm, so the gradient reaching it is
exactly what that branch sends back into the residual stream. Logged per block:
    branch/block NN/grad cos         mean over tokens of cos(g_attention, g_mlp)
    branch/block NN/grad norm ratio  mean ||g_attention|| / mean ||g_mlp||
(The loss scale multiplies both gradients, so neither metric depends on it.)

BranchProbe (fixed probe batch, eval mode):
    branch/probe CE loss
    branch/block NN/{attention, mlp} loss increase   CE loss with the branch output (+ bias) zeroed,
                                                     minus the probe CE loss
    branch/block NN/jacobian {write, read} overlap   overlap of the top-k left (write) / right (read)
                                                     singular subspaces of the two branches'
                                                     per-token Jacobians w.r.t. their residual input

The Jacobians are taken at one position per probe sequence (branch_probe_sequences sequences), so no
two tokens interact through attention: d x d each, built from d vector-Jacobian products of the branch
recomputed on its captured input. Overlap is ||Ua^T Um||_F^2 / k with k = max(1, d // 8), as
activation_geometry's subspace_overlap: 1 = same subspace, about k / d for random ones.

Assumes model_parallel_size == 1, no pipeline parallelism, no activation checkpointing.
"""

import torch

from megatron.numeric_stats import _first_tensor

BRANCHES = ("attention", "mlp")
NORMS = {"attention": "input_layernorm", "mlp": "post_attention_layernorm"}


def transformer_layers(model, neox_args, what):
    from megatron import mpu
    from megatron.model.transformer import ParallelTransformerLayerPipe

    assert mpu.get_model_parallel_world_size() == 1, f"{what} assumes model_parallel_size == 1"
    assert not neox_args.is_pipe_parallel, f"{what} assumes no pipeline parallelism"
    assert not neox_args.checkpoint_activations, f"{what} assumes no activation checkpointing"
    layers = [m for m in model.module.modules() if isinstance(m, ParallelTransformerLayerPipe)]
    assert layers and all(hasattr(l, NORMS["mlp"]) for l in layers), (
        f"{what} expects transformer layers with input_layernorm and post_attention_layernorm"
    )
    return layers


def _mean_over_ranks(values):
    values = torch.stack(values).float()
    torch.distributed.all_reduce(values)
    return values / torch.distributed.get_world_size()


class BranchGradSimilarity:
    """
        grads.start()   # before the micro-batch's forward
        grads.stop()    # after it (the gradients arrive in its backward)
        metrics = grads.compute()   # after the backward: {name: 0-dim tensor}, mean over ranks
    """

    def __init__(self, model, neox_args):
        self.layers = transformer_layers(model, neox_args, "branch_grad_interval")
        self.active = False
        self.grads = {}
        for layer in self.layers:
            for branch in BRANCHES:
                getattr(layer, NORMS[branch]).register_forward_pre_hook(self._pre_hook(layer.layer_number, branch))

    def _pre_hook(self, number, branch):
        def hook(_module, args):
            x = args[0]
            if not (self.active and torch.is_grad_enabled() and x.requires_grad):
                return None
            x = x.view_as(x)

            def save(grad):
                self.grads[(number, branch)] = grad.detach()

            x.register_hook(save)
            return (x,) + tuple(args[1:])

        return hook

    def start(self):
        self.grads = {}
        self.active = True

    def stop(self):
        self.active = False

    @torch.no_grad()
    def compute(self):
        names, values = [], []
        for layer in self.layers:
            n = layer.layer_number
            if (n, "attention") not in self.grads or (n, "mlp") not in self.grads:
                raise RuntimeError(f"branch grads: no gradients captured for block {n:02d}")
            ga, gm = (self.grads[(n, b)].float() for b in BRANCHES)
            na, nm = ga.norm(dim=-1), gm.norm(dim=-1)
            names += [f"branch/block {n:02d}/grad cos", f"branch/block {n:02d}/grad norm ratio"]
            values += [((ga * gm).sum(-1) / (na * nm).clamp_min(1e-30)).mean(), na.mean() / nm.mean()]
        self.grads = {}
        return dict(zip(names, _mean_over_ranks(values)))


def jacobians(fn, r, positions):
    """[n, d_out, d_in] Jacobians of fn(r)[p_j, j] w.r.t. r[p_j, j], for r [s, n, d] and positions [n]."""
    n, d = len(positions), r.shape[-1]
    seqs = torch.arange(n, device=r.device)
    r = r.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        y = fn(r)[positions, seqs]  # [n, d]; sequences don't interact, so row j depends on r[:, j] only
        rows = [
            torch.autograd.grad(y[:, i].sum(), r, retain_graph=i < d - 1)[0][positions, seqs].float()
            for i in range(d)
        ]
    return torch.stack(rows, dim=1)


def subspace_overlaps(ja, jm):
    """Mean over tokens of the top-k left (write) and right (read) singular-subspace overlaps."""
    k = max(1, ja.shape[-1] // 8)
    ua, _, vha = torch.linalg.svd(ja)
    um, _, vhm = torch.linalg.svd(jm)
    write = (ua[..., :k].transpose(-1, -2) @ um[..., :k]).pow(2).sum((-1, -2)) / k
    read = (vha[..., :k, :] @ vhm[..., :k, :].transpose(-1, -2)).pow(2).sum((-1, -2)) / k
    return write.mean(), read.mean()


class BranchProbe:
    """
        metrics = probe.run(model, forward)   # forward() runs the probe batch, returns its CE loss
    """

    def __init__(self, model, neox_args):
        self.layers = transformer_layers(model, neox_args, "branch_probe_interval")
        self.n_seqs = neox_args.branch_probe_sequences
        self.seed = neox_args.seed
        self.ablate = None  # (layer number, branch) whose output is zeroed
        self.capturing = False
        self.inputs = {}
        for layer in self.layers:
            n = layer.layer_number
            layer.register_forward_pre_hook(self._mask_hook(n))
            for branch in BRANCHES:
                getattr(layer, branch).register_forward_hook(self._ablate_hook(n, branch))
                getattr(layer, NORMS[branch]).register_forward_pre_hook(self._capture_hook(n, branch))

    def _mask_hook(self, n):
        def hook(_module, args):  # ParallelTransformerLayerPipe gets ((hidden_states, attention_mask),)
            if self.capturing:
                self.inputs[(n, "mask")] = args[0][1]

        return hook

    def _capture_hook(self, n, branch):
        def hook(_module, args):
            if self.capturing:
                self.inputs[(n, branch)] = args[0][:, : self.n_seqs].detach()

        return hook

    def _ablate_hook(self, n, branch):
        def hook(_module, _inputs, output):
            if self.ablate != (n, branch):
                return None
            out, bias = output
            if isinstance(out, (list, tuple)):  # [output, present] when caching keys/values
                out = type(out)([torch.zeros_like(out[0])] + list(out[1:]))
            else:
                out = torch.zeros_like(out)
            return out, None if bias is None else torch.zeros_like(bias)

        return hook

    def _branch_fn(self, layer, branch):
        norm = getattr(layer, NORMS[branch])
        if branch == "attention":
            mask = self.inputs[(layer.layer_number, "mask")]
            return lambda r: _first_tensor(layer.attention(norm(r), mask), add_bias=True)
        return lambda r: _first_tensor(layer.mlp(norm(r)), add_bias=True)

    def run(self, model, forward):
        was_training = model.training
        model.eval()
        try:
            self.inputs = {}
            self.capturing = True
            with torch.no_grad():
                base = forward()
            self.capturing = False
            names, values = ["branch/probe CE loss"], [base]
            for layer in self.layers:
                for branch in BRANCHES:
                    self.ablate = (layer.layer_number, branch)
                    with torch.no_grad():
                        loss = forward()
                    names.append(f"branch/block {layer.layer_number:02d}/{branch} loss increase")
                    values.append(loss - base)
            self.ablate = None
            r = self.inputs[(self.layers[0].layer_number, "attention")]
            generator = torch.Generator().manual_seed(self.seed)
            positions = torch.randint(r.shape[0], (r.shape[1],), generator=generator).to(r.device)
            for layer in self.layers:
                n = layer.layer_number
                ja, jm = (
                    jacobians(self._branch_fn(layer, b), self.inputs[(n, b)], positions) for b in BRANCHES
                )
                write, read = subspace_overlaps(ja, jm)
                names += [f"branch/block {n:02d}/jacobian write overlap", f"branch/block {n:02d}/jacobian read overlap"]
                values += [write, read]
        finally:
            self.ablate, self.capturing, self.inputs = None, False, {}
            model.train(was_training)
        return dict(zip(names, _mean_over_ranks(values)))
