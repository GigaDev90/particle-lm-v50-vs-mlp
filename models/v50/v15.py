"""V8: calibrated full-width particle routing with random resident initialization.

The default query projection is 512 -> 512, reshaped into four 128-dimensional
routing heads. This removes V5's 512 -> 64 projection bottleneck. Preserving
the projected dimension does not guarantee invertibility: the learned matrix
may be singular, and each head's unit normalization discards its magnitude.
Resident count adjusts automatically to keep the original parameter budget.
Support radius, kernel gain, and response initialization gain were calibrated
on synthetic initialization statistics and frozen before text training. The
response gain changes only the down-projection initialization standard
deviation; there is no permanent output multiplier. Resident initialization
may use a rotated regular simplex without changing raw norms or RNG state.

Each resident belongs to one routing head and has a spherical position, radius,
and a bank of full-width content responses. One compact radial gate controls
that resident's entire bank. Broadcast multiplication avoids constructing a
full FP32 routing tensor for every response channel. Residents outside compact
support contribute exactly zero, with no content path around the radial gate.
The explicit uniform-gate ablation preserves the parameter budget but leaves
the geometry parameters unused; it is not a particle-routing result.
"""

from dataclasses import asdict, dataclass, replace
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 8192
    context_length: int = 512
    n_layers: int = 8
    width: int = 512
    n_heads: int = 8
    n_routing_heads: int = 4
    routing_dim: int = 128
    response_channels: int = 4
    dropout: float = 0.0
    bias: bool = True
    target_parameters: int = 29_676_544
    n_particles: int | None = None
    radius_squared: float = 2.171848773956299
    kernel_gain: float = 6.617806196212754
    response_init_gain: float = 1.0750019319619315
    resident_init: str = "random"
    gate_mode: str = "radial"
    routing_precision: str = "mixed"


def _helmert_columns(n, device):
    """An orthonormal basis of the zero-sum subspace of R^n."""
    basis = torch.zeros(n, n - 1, dtype=torch.float64, device=device)
    for column in range(n - 1):
        count = column + 1
        denominator = math.sqrt(count * (count + 1))
        basis[:count, column] = 1 / denominator
        basis[count, column] = -count / denominator
    return basis


@torch.no_grad()
def simplex_resident_positions(original_positions):
    """Return rotated simplex directions scaled by original per-resident norms.

    Orientation comes from QR(original.T @ Helmert), with QR signs chosen to
    make R's diagonal positive. For independent standard Gaussian residents,
    this gives a rotationally invariant orthonormal frame without an RNG call.
    Every leading batch item (e.g. layer/head) receives its own frame.

    The result is detached and intended for copying into an existing trainable
    parameter or wrapping in nn.Parameter. Input shape, dtype, and device are
    preserved; input values and all random-number-generator states are untouched.
    Geometry and norms hold to the returned dtype's numerical precision.

    A regular simplex requires 2 <= N <= D+1. Nonfinite values, zero norms, or
    rank-deficient/near-degenerate orientation inputs are rejected explicitly.
    There is no fallback random draw for degenerate inputs.
    """
    if not isinstance(original_positions, torch.Tensor):
        raise TypeError("original_positions must be a torch.Tensor")
    if original_positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("original_positions must use float32 or float64")
    if original_positions.ndim < 2 or original_positions.numel() == 0:
        raise ValueError("expected a nonempty (..., residents, dimensions) tensor")
    residents, dimensions = original_positions.shape[-2:]
    if dimensions < 1 or not 2 <= residents <= dimensions + 1:
        raise ValueError("regular simplex requires 2 <= residents <= dimensions + 1")
    if not torch.isfinite(original_positions).all():
        raise ValueError("resident positions must be finite")
    raw = original_positions.double()
    norms = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    if not torch.isfinite(norms).all() or not (norms > 0).all():
        raise ValueError("each original resident must have a finite, nonzero norm")

    helmert = _helmert_columns(residents, original_positions.device)
    orientation_source = raw.transpose(-1, -2) @ helmert
    singular_values = torch.linalg.svdvals(orientation_source)
    relative_tolerance = max(dimensions, residents - 1) * torch.finfo(original_positions.dtype).eps
    if (singular_values[..., -1] <= relative_tolerance * singular_values[..., 0]).any():
        raise ValueError("original draws give a rank-deficient or near-degenerate orientation")
    frame, triangular = torch.linalg.qr(orientation_source, mode="reduced")
    signs = torch.where(torch.diagonal(triangular, dim1=-2, dim2=-1) < 0, -1.0, 1.0)
    frame = frame * signs.unsqueeze(-2)
    vertices = helmert * math.sqrt(residents / (residents - 1))
    directions = vertices @ frame.transpose(-1, -2)
    return (directions * norms).to(dtype=original_positions.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_width = config.width // config.n_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=config.bias)
        self.projection = nn.Linear(config.width, config.width, bias=config.bias)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.view(batch, length, self.n_heads, self.head_width).transpose(1, 2)
            for t in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.residual_dropout(self.projection(attended))


class ParticleField(nn.Module):
    """One spherical neighborhood gates a bank of full-width content responses.

    q_a = normalize(Wq_a h), p_ai = normalize(raw_position_ai)
    d_ai^2 = ||q_a||^2 + ||p_ai||^2 - 2 q_a dot p_ai
    w_ai = kernel_gain * relu(1 - d_ai^2 / radius_ai^2)^2
    kappa_ai = w_ai / sqrt(mean_j(w_aj^2) + 1/(M*C))
    F(h) = sum_ai kappa_ai * sum_c SiLU(u_aic dot h + b_aic) * v_aic

    M is the number of residents per head; C is response channels per resident.
    The floor 1/(M*C) exactly matches a conceptual channel-wise model in which
    each resident's gate is repeated C times before normalization. The actual
    implementation broadcasts resident gates without constructing that larger
    FP32 gate tensor. C=1 with both gains set to one recovers the v3 law.
    These weights are nonnegative kernel values, not probabilities.
    Norms retain their actual values,
    including zero, rather than assuming every normalized vector has norm one.
    Unit normalization, norms, radii, kernels, and gate normalization use FP32.
    In mixed mode, query projection and the batched routing dot product use the
    ambient autocast precision; fp32 mode provides a full-precision reference.
    Every content response uses all input/output coordinates. No extra head
    averaging, output bias, or route around the compact kernel is present.
    """

    def __init__(self, config):
        super().__init__()
        if config.n_particles is None or config.n_particles < 1:
            raise ValueError("ParticleField requires a positive n_particles")
        if min(config.routing_dim, config.n_routing_heads, config.response_channels) < 1:
            raise ValueError("routing dimensions and response_channels must be positive")
        if config.n_particles % config.n_routing_heads:
            raise ValueError("n_particles must be divisible by n_routing_heads")
        for name in ("radius_squared", "kernel_gain", "response_init_gain"):
            value = getattr(config, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if config.resident_init not in ("random", "simplex"):
            raise ValueError("resident_init must be random or simplex")
        if config.resident_init == "simplex" and not 2 <= config.n_particles // config.n_routing_heads <= config.routing_dim + 1:
            raise ValueError("simplex requires 2 <= residents per head <= routing_dim + 1")
        if config.gate_mode not in ("radial", "uniform"):
            raise ValueError("gate_mode must be radial or uniform")
        if config.routing_precision not in ("mixed", "fp32"):
            raise ValueError("routing_precision must be mixed or fp32")
        self.width = config.width
        self.routing_dim = config.routing_dim
        self.n_routing_heads = config.n_routing_heads
        self.n_particles = config.n_particles
        self.particles_per_head = config.n_particles // config.n_routing_heads
        self.response_channels = config.response_channels
        self.total_response_channels = config.n_particles * config.response_channels
        self.gate_mode = config.gate_mode
        self.routing_precision = config.routing_precision
        self.kernel_gain = config.kernel_gain
        self.response_init_gain = config.response_init_gain
        self.resident_init = config.resident_init
        self.query_projection = nn.Linear(
            config.width, config.n_routing_heads * config.routing_dim, bias=False
        )
        self.raw_positions = nn.Parameter(torch.empty(
            config.n_routing_heads, self.particles_per_head, config.routing_dim
        ))
        self.log_radius_squared = nn.Parameter(torch.full(
            # Keep this vector flat so the existing optimizer's dimension-based
            # no-decay rule treats radii exactly as it did in v2.
            (config.n_particles,), math.log(config.radius_squared)
        ))
        self.up = nn.Linear(config.width, self.total_response_channels, bias=config.bias)
        self.down = nn.Linear(self.total_response_channels, config.width, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        nn.init.normal_(self.raw_positions, std=1.0)
        if self.resident_init == "simplex":
            with torch.no_grad():
                self.raw_positions.copy_(simplex_resident_positions(self.raw_positions))
        nn.init.normal_(self.query_projection.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        nn.init.normal_(self.down.weight, std=config.response_init_gain * 0.02 / math.sqrt(2 * config.n_layers))

    @property
    def positions(self):
        return F.normalize(self.raw_positions.float(), dim=-1, eps=1e-6)

    @property
    def radius_squared(self):
        return self.log_radius_squared.float().exp().reshape(self.n_routing_heads, self.particles_per_head)

    def routing_query(self, h):
        if self.routing_precision == "fp32":
            with torch.autocast(device_type=h.device.type, enabled=False):
                projected = F.linear(h.float(), self.query_projection.weight.float())
        else:
            projected = self.query_projection(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            projected = projected.float().reshape(*h.shape[:-1], self.n_routing_heads, self.routing_dim)
            return F.normalize(projected, dim=-1, eps=1e-6)

    def routing_distance_squared(self, h):
        """Return distances in flattened head/resident order, like content units."""
        if h.shape[-1] != self.width:
            raise ValueError("input's final dimension must equal particle width")
        q = self.routing_query(h).reshape(-1, self.n_routing_heads, self.routing_dim).transpose(0, 1)
        with torch.autocast(device_type=h.device.type, enabled=False):
            p = self.positions
            query_norms = q.square().sum(dim=-1, keepdim=True)
            resident_norms = p.square().sum(dim=-1).unsqueeze(1)
        if self.routing_precision == "fp32":
            with torch.autocast(device_type=h.device.type, enabled=False):
                dot = torch.bmm(q, p.transpose(1, 2))
        else:
            dot = torch.bmm(q, p.transpose(1, 2))
        with torch.autocast(device_type=h.device.type, enabled=False):
            distances = (query_norms + resident_norms - 2.0 * dot.float()).clamp_min(0.0)
        return distances.transpose(0, 1).reshape(*h.shape[:-1], self.n_particles)

    def interaction_weights(self, h):
        if h.shape[-1] != self.width:
            raise ValueError("input's final dimension must equal particle width")
        if self.gate_mode == "uniform":
            return torch.ones((*h.shape[:-1], self.n_particles), device=h.device, dtype=torch.float32)
        distance_squared = self.routing_distance_squared(h)
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.kernel_gain * F.relu(1.0 - distance_squared / self.radius_squared.reshape(-1)).square()

    def normalized_weights(self, weights):
        with torch.autocast(device_type=weights.device.type, enabled=False):
            heads = weights.float().reshape(*weights.shape[:-1], self.n_routing_heads, self.particles_per_head)
            gates = heads * torch.rsqrt(
                heads.square().mean(dim=-1, keepdim=True)
                + 1.0 / (self.particles_per_head * self.response_channels)
            )
            return gates.reshape_as(weights)

    def forward(self, h):
        weights = self.normalized_weights(self.interaction_weights(h))
        content = F.silu(self.up(h))
        content = content.reshape(
            *h.shape[:-1], self.n_routing_heads, self.particles_per_head, self.response_channels
        )
        resident_gates = weights.to(content.dtype).reshape(
            *h.shape[:-1], self.n_routing_heads, self.particles_per_head, 1
        )
        responses = (content * resident_gates).flatten(-3)
        return self.dropout(self.down(responses))

    @torch.no_grad()
    def diagnostics(self, h, force=None):
        flat = h.reshape(-1, self.width)[:256]
        if not flat.numel():
            raise ValueError("diagnostics requires at least one query")
        weights = self.interaction_weights(flat)
        gates = self.normalized_weights(weights)
        sums_squared = weights.sum(dim=-1).square()
        square_sums = weights.square().sum(dim=-1)
        effective_neighbors = sums_squared / square_sums.clamp_min(1e-12)
        if force is None:
            force = self(flat)
        else:
            force = force.reshape(-1, self.width)[:flat.shape[0]]
        force_rms = force.float().square().mean().sqrt().item()
        unit_weights = F.normalize(gates, dim=-1, eps=1e-12)
        count = unit_weights.shape[0]
        if count > 1:
            pair_cosine = (
                unit_weights.sum(dim=0).square().sum()
                - unit_weights.square().sum()
            ) / (count * (count - 1))
        else:
            pair_cosine = weights.new_zeros(())
        r2 = self.radius_squared
        head_weights = weights.reshape(-1, self.n_routing_heads, self.particles_per_head)
        head_effective_neighbors = (
            head_weights.sum(dim=-1).square()
            / head_weights.square().sum(dim=-1).clamp_min(1e-12)
        )
        return {
            "gate_mode": self.gate_mode,
            "routing_precision": self.routing_precision,
            "kernel_gain": self.kernel_gain,
            "response_init_gain": self.response_init_gain,
            "resident_init": self.resident_init,
            "routing_heads": self.n_routing_heads,
            "routing_dim": self.routing_dim,
            "particles_per_head": self.particles_per_head,
            "residents_per_head": self.particles_per_head,
            "total_residents": self.n_particles,
            "response_channels_per_resident": self.response_channels,
            "response_channels_per_head": self.particles_per_head * self.response_channels,
            "total_response_channels": self.total_response_channels,
            "active_fraction": (weights > 0).float().mean().item(),
            "empty_query_fraction": (~(weights > 0).any(dim=-1)).float().mean().item(),
            "effective_neighbors": effective_neighbors.mean().item(),
            "mean_effective_neighbors_per_head": head_effective_neighbors.mean().item(),
            "effective_response_channels": (self.response_channels * effective_neighbors.mean()).item(),
            "mean_effective_response_channels_per_head": (
                self.response_channels * head_effective_neighbors.mean()
            ).item(),
            "mean_active_residents_per_head": (head_weights > 0).float().sum(dim=-1).mean().item(),
            "empty_head_fraction": (~(head_weights > 0).any(dim=-1)).float().mean().item(),
            "gate_rms": gates.square().mean().sqrt().item(),
            "mean_gate_cosine": pair_cosine.item(),
            "query_rms": flat.float().square().mean().sqrt().item(),
            "force_rms": force_rms,
            "trajectory_displacement_rms": force_rms,
            "radius_squared": r2.mean().item(),
            "radius_squared_min": r2.min().item(),
            "radius_squared_max": r2.max().item(),
        }


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.width, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.particle_norm = nn.LayerNorm(config.width, bias=config.bias)
        self.field = ParticleField(config)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.field(self.particle_norm(x))


class GPT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = replace(config) if config is not None else ModelConfig()
        c = self.config
        if min(c.vocab_size, c.context_length, c.n_layers, c.width, c.n_heads,
               c.routing_dim, c.n_routing_heads, c.response_channels) < 1:
            raise ValueError("model dimensions must be positive")
        if c.width % c.n_heads:
            raise ValueError("width must be divisible by n_heads")
        if not 0 <= c.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if c.n_particles is None:
            # A resident owns one position/radius and C pairs of full-width
            # response vectors, each with an optional content bias.
            per_particle = c.routing_dim + 1 + c.response_channels * (2 * c.width + int(c.bias))
            norm_parameters = c.width * (1 + int(c.bias))
            attention_parameters = 4 * c.width * c.width + (4 * c.width if c.bias else 0)
            fixed = (
                (c.vocab_size + c.context_length) * c.width + norm_parameters
                + c.n_layers * (
                    2 * norm_parameters + attention_parameters
                    + c.width * c.n_routing_heads * c.routing_dim
                )
            )
            c.n_particles = c.n_routing_heads * round(
                (c.target_parameters - fixed)
                / (c.n_layers * per_particle * c.n_routing_heads)
            )
            if c.n_particles < 1:
                raise ValueError("target_parameters leaves no room for particle fields")
        self.token_embedding = nn.Embedding(c.vocab_size, c.width)
        self.position_embedding = nn.Embedding(c.context_length, c.width)
        self.dropout = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.final_norm = nn.LayerNorm(c.width, bias=c.bias)
        self.lm_head = nn.Linear(c.width, c.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)
        for block in self.blocks:
            nn.init.normal_(block.attention.projection.weight, std=0.02 / math.sqrt(2 * c.n_layers))
            nn.init.normal_(block.field.down.weight, std=c.response_init_gain * 0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _embed(self, tokens):
        length = tokens.size(1)
        if length > self.config.context_length:
            raise ValueError("Sequence length exceeds the configured context")
        positions = torch.arange(length, device=tokens.device)
        return self.dropout(self.token_embedding(tokens) + self.position_embedding(positions))

    def forward(self, tokens, targets=None):
        x = self._embed(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        if targets is None:
            return self.lm_head(x[:, [-1], :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def particle_diagnostics(self, tokens):
        x = self._embed(tokens)
        results = []
        for layer, block in enumerate(self.blocks):
            x = x + block.attention(block.attention_norm(x))
            h = block.particle_norm(x)
            force = block.field(h)
            results.append({"layer": layer, **block.field.diagnostics(h, force)})
            x = x + force
        return results

    def parameter_count(self):
        return parameter_count(self)

    def config_dict(self):
        return config_dict(self.config)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def config_dict(config):
    return asdict(config.config if isinstance(config, GPT) else config)
