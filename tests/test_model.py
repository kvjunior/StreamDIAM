"""Model tests: shapes, determinism, full gradient flow, exact parameter count.

These guard the two failure modes that sank the prior prototype: a reported
parameter count that did not match the code, and an MGD cache that silently
detached gradients. The gradient test asserts *every* parameter receives a
non-zero gradient; the parameter-count test asserts the model matches a closed
form computed from the dimensions.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")  # skip cleanly if torch is unavailable

from streamdiam.model import DIAM, Edge2Seq  # noqa: E402

D, C, T, N, E = 7, 16, 5, 12, 40


def make_inputs(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    in_seq = torch.randn(N, T, D, generator=g)
    out_seq = torch.randn(N, T, D, generator=g)
    in_len = torch.randint(1, T + 1, (N,), generator=g)
    out_len = torch.randint(1, T + 1, (N,), generator=g)
    src = torch.randint(0, N, (E,), generator=g)
    dst = torch.randint(0, N, (E,), generator=g)
    edge_index = torch.stack([src, dst], dim=0)
    return in_seq, in_len, out_seq, out_len, edge_index


def expected_params(d: int, c: int, num_layers: int, shared: bool) -> int:
    """Closed-form parameter count (see model docstrings for the derivation)."""
    def mgd(p_in: int) -> int:
        base = 3 * c * c + p_in * c + 3 * c  # lin_z + msg_in + att_w + att_q
        return base if shared else base + 2 * c * c  # +msg_out when not shared
    edge2seq = 12 * c * c + 2 * d * c + 14 * c     # 2 linears + 2 GRUs
    classifier = c * c + 2 * c + 1                 # Linear(c,c) + Linear(c,1)
    return edge2seq + mgd(2 * c) + (num_layers - 1) * mgd(c) + classifier


def test_forward_shape_and_finiteness():
    model = DIAM(in_dim=D, hidden_dim=C, num_layers=2).eval()
    logits = model(*make_inputs())
    assert logits.shape == (N,)
    assert torch.isfinite(logits).all()


def test_predict_proba_in_unit_interval():
    model = DIAM(in_dim=D, hidden_dim=C, num_layers=2)
    proba = model.predict_proba(*make_inputs())
    assert proba.shape == (N,)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0


def test_determinism_in_eval_mode():
    torch.manual_seed(1)
    model = DIAM(in_dim=D, hidden_dim=C, num_layers=2).eval()
    x = make_inputs(seed=3)
    a = model(*x)
    b = model(*x)
    assert torch.equal(a, b)


def test_gradient_reaches_every_parameter():
    """No parameter may be left without a (non-zero) gradient."""
    torch.manual_seed(2)
    model = DIAM(in_dim=D, hidden_dim=C, num_layers=2).train()
    x = make_inputs(seed=5)
    logits = model(*x)
    labels = (torch.arange(N) % 2).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss.backward()
    dead = []
    for name, p in model.named_parameters():
        if p.grad is None or float(p.grad.abs().sum()) == 0.0:
            dead.append(name)
    assert not dead, f"parameters with no gradient: {dead}"


@pytest.mark.parametrize("num_layers", [1, 2, 3])
@pytest.mark.parametrize("shared", [True, False])
def test_parameter_count_matches_closed_form(num_layers, shared):
    model = DIAM(in_dim=D, hidden_dim=C, num_layers=num_layers, shared_message=shared)
    assert model.num_parameters() == expected_params(D, C, num_layers, shared)


def test_shared_message_has_fewer_params_than_separate():
    shared = DIAM(in_dim=D, hidden_dim=C, num_layers=2, shared_message=True)
    separate = DIAM(in_dim=D, hidden_dim=C, num_layers=2, shared_message=False)
    assert separate.num_parameters() > shared.num_parameters()


def test_maxpool_ignores_padding():
    """Values beyond a node's sequence length must not affect its encoding."""
    torch.manual_seed(7)
    enc = Edge2Seq(in_dim=D, hidden_dim=C).eval()
    in_seq = torch.randn(N, T, D)
    out_seq = torch.randn(N, T, D)
    in_len = torch.full((N,), T, dtype=torch.long)
    out_len = torch.full((N,), T, dtype=torch.long)
    in_len[0] = 1  # node 0 uses only its first incoming edge
    h1 = enc(in_seq, in_len, out_seq, out_len)
    poisoned = in_seq.clone()
    poisoned[0, 1:, :] = 1e4  # overwrite node 0's padded region with garbage
    h2 = enc(poisoned, in_len, out_seq, out_len)
    assert torch.allclose(h1, h2, atol=1e-5)


def test_length_zero_rejected():
    enc = Edge2Seq(in_dim=D, hidden_dim=C)
    in_seq = torch.randn(N, T, D)
    out_seq = torch.randn(N, T, D)
    bad = torch.full((N,), T, dtype=torch.long)
    bad[0] = 0
    with pytest.raises(ValueError):
        enc(in_seq, bad, out_seq, torch.full((N,), T, dtype=torch.long))


def test_num_layers_validation():
    with pytest.raises(ValueError):
        DIAM(in_dim=D, hidden_dim=C, num_layers=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
