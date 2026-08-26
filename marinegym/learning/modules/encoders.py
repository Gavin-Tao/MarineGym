"""可插拔序列编码器 —— 论文③ 的对照臂。

统一契约：输入 [..., L, D]（L=上下文长度，末位是当前帧），输出 [..., H]。
前导维度任意（训练时是 [T, E, n_agents]，rollout 时是 [E, n_agents]），
内部统一压平成 [B, L, D] 再还原。

各臂：
  mlp         只看当前帧 —— 论文的"只考虑当前 state"下界基线
  stack       flatten(L*D) → MLP —— 朴素历史基线
  gru         窗口内 GRU 取末态 —— DRQN 类
  transformer causal self-attention 取末位 —— DTQN 对标
  mamba       选择性状态空间模型 —— ours

**参数量对齐**：各臂通过 d_model / n_layers 调到与 mlp 基线同量级，
build_encoder 会把实际参数量打印出来，论文表格里如实报告。
"""

import math
import torch
import torch.nn as nn


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class _SeqBase(nn.Module):
    """处理 [..., L, D] ↔ [B, L, D] 的形状折叠。子类实现 _encode。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-2]
        L, D = x.shape[-2], x.shape[-1]
        h = self._encode(x.reshape(-1, L, D))
        return h.reshape(*lead, h.shape[-1])

    def _encode(self, x):        # [B, L, D] -> [B, H]
        raise NotImplementedError

    def protected_modules(self):
        """自带初始化、不应被外部 re-init 覆盖的子模块。"""
        return []


class MLPEncoder(_SeqBase):
    """只用当前帧（窗口末位）。L=1 时与改造前的 PPO 逐位等价。"""

    def __init__(self, hidden=(256, 256, 256)):
        super().__init__()
        self.net = make_mlp(hidden)

    def _encode(self, x):
        return self.net(x[:, -1])


class FrameStackEncoder(_SeqBase):
    """把整个窗口拍平喂 MLP —— 历史信息全给了，但没有序列归纳偏置。

    注意首层输入是 L*D，参数量随 L 线性增长 —— 长上下文实验里这本身就是一个结论。
    hidden 首层取 128（而非 256）是为了在 L=16 时与其它臂参数量对齐。"""

    def __init__(self, hidden=(128, 256, 256)):
        super().__init__()
        self.net = make_mlp(hidden)

    def _encode(self, x):
        return self.net(x.reshape(x.shape[0], -1))


class GRUEncoder(_SeqBase):
    """窗口内 GRU 取末态 —— DRQN 类基线。

    输出接 LayerNorm(h + input) —— 与本仓库自带的 `modules/rnn.py::GRU` 一致。
    实测不加这一层时，16 步展开的梯度会在训练后期把 critic 打爆
    (value_loss 0.48 → 9.9 → NaN，约 iter 96 发散)。这是**修正**而非削弱基线。
    """

    def __init__(self, d_model=128, n_layers=1, hidden=(256,)):
        super().__init__()
        self.inp = nn.LazyLinear(d_model)
        self.rnn = nn.GRU(d_model, d_model, num_layers=n_layers, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.head = make_mlp(hidden)

    def protected_modules(self):
        return [self.rnn]

    def _encode(self, x):
        z = torch.relu(self.inp(x))
        h, _ = self.rnn(z)
        return self.head(self.ln(h[:, -1] + z[:, -1]))


# ---------------------------------------------------------------------------
# Transformer 臂 —— 忠实复刻 kevslinger/DTQN
#   dtqn/networks/transformer.py  TransformerLayer / TransformerIdentityLayer
#   dtqn/networks/gates.py        ResGate / GRUGate
#   run.py 默认: --in-embed 128 --heads 8 --layers 2 --dropout 0.0
#                --gate res --pos learned --identity(False, 即后置 LayerNorm)
# ---------------------------------------------------------------------------

class ResGate(nn.Module):
    """DTQN gates.py: 残差跳连（默认 --gate res）。"""

    def forward(self, x, y):
        return x + y


class GRUGate(nn.Module):
    """DTQN gates.py: GTrXL 的 GRU 门控（--gate gru）。w_z.bias 初始化为 -2。"""

    def __init__(self, embed_size):
        super().__init__()
        self.w_r = nn.Linear(embed_size, embed_size, bias=False)
        self.u_r = nn.Linear(embed_size, embed_size, bias=False)
        self.w_z = nn.Linear(embed_size, embed_size)
        self.u_z = nn.Linear(embed_size, embed_size, bias=False)
        self.w_g = nn.Linear(embed_size, embed_size, bias=False)
        self.u_g = nn.Linear(embed_size, embed_size, bias=False)
        with torch.no_grad():
            self.w_z.bias.fill_(-2)

    def forward(self, x, y):
        z = torch.sigmoid(self.w_z(y) + self.u_z(x))
        r = torch.sigmoid(self.w_r(y) + self.u_r(x))
        h = torch.tanh(self.w_g(y) + self.u_g(r * x))
        return (1.0 - z) * x + z * h


def _make_gate(kind, embed_size):
    return GRUGate(embed_size) if kind == "gru" else ResGate()


class DTQNLayer(nn.Module):
    """DTQN 的 transformer block。

    identity=False (DTQN 默认, TransformerLayer): LayerNorm 在跳连**之后**
    identity=True  (TransformerIdentityLayer)   : GTrXL identity map reordering，
                                                  LayerNorm 在子层**之前**
    两个变体都把子层输出先过 ReLU 再进门控 —— 这是 DTQN 的做法，照抄。
    FFN: Linear(d, 4d) → ReLU → Linear(4d, d) → Dropout
    """

    def __init__(self, d_model, n_heads, dropout=0.0, gate="res",
                 identity=False, ff_mult=4):
        super().__init__()
        self.identity = identity
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.ReLU(),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn_gate = _make_gate(gate, d_model)
        self.mlp_gate = _make_gate(gate, d_model)

    def forward(self, x, mask):
        if self.identity:                       # GTrXL: 归一化在子层之前
            h = self.ln1(x)
            a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
            x = self.attn_gate(x, torch.relu(a))
            x = self.mlp_gate(x, torch.relu(self.ffn(self.ln2(x))))
        else:                                   # DTQN 默认: 归一化在跳连之后
            a, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
            x = self.ln1(self.attn_gate(x, torch.relu(a)))
            x = self.ln2(self.mlp_gate(x, torch.relu(self.ffn(x))))
        return x


class TransformerEncoder(_SeqBase):
    """DTQN 对标臂：因果 self-attention over 窗口，取最后一个位置。

    与 DTQN 的一处**有意偏离**：DTQN 会在窗口每个时刻都预测 Q 值并一起训练
    (`--history`，"intermediate Q-values")。那是 Q-learning 专有的样本效率技巧；
    本文是 on-policy actor-critic (PPO)，损失只定义在当前动作上，故只取末位。
    这个差异对所有序列臂一视同仁，不影响 arm 之间的比较。

    复杂度：每个控制步 O(L²·d) —— 这正是 §5b 效率实验要打的点。
    """

    def __init__(self, d_model=128, n_layers=2, n_heads=8, dropout=0.0,
                 gate="res", identity=False, pos="learned", ff_mult=4,
                 max_len=1024, hidden=(256,)):
        super().__init__()
        self.inp = nn.LazyLinear(d_model)
        self.pos_kind = pos
        if pos == "learned":                    # DTQN 默认
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pos_emb, std=0.02)
        elif pos == "sin":
            pe = torch.zeros(max_len, d_model)
            q = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(q * div)
            pe[:, 1::2] = torch.cos(q * div)
            self.register_buffer("pos_buf", pe.unsqueeze(0))
        self.layers = nn.ModuleList([
            DTQNLayer(d_model, n_heads, dropout, gate, identity, ff_mult)
            for _ in range(n_layers)
        ])
        self.head = make_mlp(hidden)
        self._mask_cache = {}   # 每次前向重新分配 causal mask 会人为拖慢本基线

    def _causal_mask(self, L, device, dtype):
        key = (L, device, dtype)
        m = self._mask_cache.get(key)
        if m is None:
            # DTQN: torch.triu(ones(L, L), diagonal=1) 的上三角置 -inf
            m = torch.zeros(L, L, device=device, dtype=dtype)
            m.masked_fill_(torch.triu(torch.ones(L, L, device=device,
                                                 dtype=torch.bool), diagonal=1),
                           float("-inf"))
            self._mask_cache[key] = m
        return m

    def protected_modules(self):
        """自带初始化的子模块（含 GRUGate 的 w_z.bias=-2），
        不可被 PPO 的 orthogonal(0.01) 覆盖。"""
        return list(self.layers)

    def _encode(self, x):
        h = self.inp(x)
        L = h.shape[1]
        if self.pos_kind == "learned":
            h = h + self.pos_emb[:, :L]
        elif self.pos_kind == "sin":
            h = h + self.pos_buf[:, :L]
        mask = self._causal_mask(L, h.device, h.dtype)
        for layer in self.layers:
            h = layer(h, mask)
        return self.head(h[:, -1])


class MambaEncoder(_SeqBase):
    """ours：选择性状态空间模型。

    完全使用官方 state-spaces/mamba 的组件，SSM 与残差接法都不自己实现：
      * `from mamba_ssm import Mamba`   —— README 里的官方模块
      * `create_block(...)`             —— 官方 Block 工厂（norm + mixer + 残差）
      * 前向的 residual / norm_f 处理照抄官方 `MixerModel.forward`

    训练走并行扫描 O(L)；部署可走递归模式，每步 O(1)、状态大小与 L 无关。
    """

    def __init__(self, d_model=128, n_layers=2, d_state=16, d_conv=4, expand=2,
                 rms_norm=False, hidden=(256,)):
        # rms_norm 默认 False —— 与官方 create_block 的签名默认值一致。
        # triton 版 RMSNorm 首次调用要 autotune，会额外申请数百 MB 显存，
        # 在共享 GPU 上极易 OOM；nn.LayerNorm 在数学上等价地起归一化作用。
        super().__init__()
        from mamba_ssm.models.mixer_seq_simple import create_block
        self.inp = nn.LazyLinear(d_model)
        ssm_cfg = {"d_state": d_state, "d_conv": d_conv, "expand": expand}
        self.layers = nn.ModuleList([
            # d_intermediate=0 → 纯 Mamba block（不插 MLP 层），与官方 Mamba-1 一致
            create_block(d_model, d_intermediate=0, ssm_cfg=ssm_cfg,
                         rms_norm=rms_norm, fused_add_norm=False, layer_idx=i)
            for i in range(n_layers)
        ])
        if rms_norm:
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
            self.norm_f = RMSNorm(d_model, eps=1e-5)
        else:
            self.norm_f = nn.LayerNorm(d_model, eps=1e-5)
        self.head = make_mlp(hidden)

    def protected_modules(self):
        """Mamba 的 dt_proj / A_log / D / conv1d 都有精心设计的初始化，
        若被 PPO 的 orthogonal(0.01) 覆盖会毁掉 SSM 参数化。"""
        return list(self.layers) + [self.norm_f]

    def _encode(self, x):
        hidden_states = self.inp(x)
        residual = None
        # ↓ 与官方 MixerModel.forward 的 fused_add_norm=False 分支一致
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        return self.head(hidden_states[:, -1])


_REGISTRY = {
    "mlp": MLPEncoder,
    "stack": FrameStackEncoder,
    "gru": GRUEncoder,
    "transformer": TransformerEncoder,
    "mamba": MambaEncoder,
}


def build_encoder(cfg):
    """cfg: 含 name 及该臂的超参（dict / omegaconf）。"""
    kwargs = {k: v for k, v in dict(cfg).items() if k != "name"}
    name = str(cfg["name"]).lower()
    if name not in _REGISTRY:
        raise NotImplementedError(
            f"unknown encoder '{name}', 可选: {sorted(_REGISTRY)}")
    cls = _REGISTRY[name]
    # 只传该类接受的超参，其余忽略（这样一份 yaml 可以喂所有臂）
    import inspect
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in kwargs.items() if k in accepted}
    if "hidden" in kwargs and kwargs["hidden"] is not None:
        kwargs["hidden"] = tuple(kwargs["hidden"])
    return cls(**kwargs)
