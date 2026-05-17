"""Sanity tests for FrequencyDecoupledDistillLoss.

Standalone script (no pytest dependency). Run from repo root:

    .venv/bin/python tests/test_fdkd.py

Covers:
  1. DCT-II round-trip (Dh.T @ DCT(x) @ Dw ≈ x).
  2. Constant input → only DC coefficient is nonzero.
  3. loss(x, x) == 0 for arbitrary logits.
  4. Gradient flows to student, NOT to teacher.
  5. Cache keyed by (N, device, dtype), no thrash on repeat calls.
  6. lf_size clamp when lf_size > min(H, W).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cellvit_distill.utils.losses import FrequencyDecoupledDistillLoss


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not cond:
        raise AssertionError(name)


def test_dct_roundtrip() -> None:
    """Orthonormal DCT-II is its own inverse via transpose."""
    print("\n[1] DCT-II round-trip")
    loss = FrequencyDecoupledDistillLoss()
    for N in (8, 32, 256):
        D = loss._dct_matrix(N, torch.device("cpu"), torch.float64)
        x = torch.randn(2, 3, N, N, dtype=torch.float64)
        # x_recon = D^T (D x D^T) D
        x_dct = torch.einsum("ij,...jk->...ik", D, x)
        x_dct = torch.einsum("...ik,jk->...ij", x_dct, D)
        x_recon = torch.einsum("ji,...jk->...ik", D, x_dct)
        x_recon = torch.einsum("...ik,kj->...ij", x_recon, D)
        err = (x - x_recon).abs().max().item()
        _check(f"N={N} max|x − D⁻¹DCT(x)|", err < 1e-9, f"max_err={err:.2e}")


def test_constant_input_dc_only() -> None:
    """Constant input → all DCT coefficients zero except (0,0)."""
    print("\n[2] Constant input → DC-only")
    loss = FrequencyDecoupledDistillLoss()
    x = torch.full((1, 1, 32, 32), 0.5, dtype=torch.float64)
    X = loss._dct2(x)
    dc = X[0, 0, 0, 0].item()
    X[0, 0, 0, 0] = 0.0
    rest_max = X.abs().max().item()
    _check("DC nonzero", abs(dc) > 0.1, f"DC={dc:.4f}")
    _check("non-DC zero", rest_max < 1e-9, f"max|non-DC|={rest_max:.2e}")


def test_self_distill_zero() -> None:
    """loss(x, x) should be exactly 0 for any logits."""
    print("\n[3] loss(x, x) == 0")
    loss = FrequencyDecoupledDistillLoss(temperature=10.0, lf_size=8)
    logits = torch.randn(2, 6, 32, 32)
    val = loss(logits, logits).item()
    _check("self-distill", val == 0.0, f"loss={val:.2e}")
    _check("last_lf_raw stashed",
           loss.last_lf_raw == 0.0, f"={loss.last_lf_raw:.2e}")
    _check("last_hf_raw stashed",
           loss.last_hf_raw == 0.0, f"={loss.last_hf_raw:.2e}")


def test_gradient_flow() -> None:
    """Gradient should flow to student, not teacher."""
    print("\n[4] Gradient flow")
    loss = FrequencyDecoupledDistillLoss(temperature=10.0, lf_size=4)
    s = torch.randn(2, 6, 16, 16, requires_grad=True)
    t = torch.randn(2, 6, 16, 16, requires_grad=False)
    val = loss(s, t)
    val.backward()
    _check("student grad exists", s.grad is not None)
    _check("student grad nonzero", s.grad.abs().sum().item() > 0)
    _check("teacher grad None", t.grad is None)


def test_cache_key() -> None:
    """Cache should not thrash on repeated same-key calls."""
    print("\n[5] DCT cache (N, device, dtype) key")
    loss = FrequencyDecoupledDistillLoss()
    dev = torch.device("cpu")
    D1 = loss._dct_matrix(64, dev, torch.float32)
    D2 = loss._dct_matrix(64, dev, torch.float32)
    _check("same call returns same tensor", D1 is D2)
    D3 = loss._dct_matrix(64, dev, torch.float64)
    _check("different dtype → different cache entry", D1 is not D3)
    keys = set(loss._dct_cache.keys())
    _check("cache holds 2 keys",
           len(keys) == 2, f"keys={keys}")


def test_lf_size_clamp() -> None:
    """lf_size > min(H, W) should clamp silently, not crash."""
    print("\n[6] lf_size clamp on small heads")
    loss = FrequencyDecoupledDistillLoss(temperature=4.0, lf_size=64)
    # 32×32 head < lf_size=64. Expect clamp to K=32 → hf_loss=0 (no HF band).
    s = torch.randn(1, 3, 32, 32)
    t = torch.randn(1, 3, 32, 32)
    val = loss(s, t).item()
    _check("forward runs without error", True)
    _check("hf_raw == 0 (no HF band when K==H==W)",
           loss.last_hf_raw == 0.0, f"hf_raw={loss.last_hf_raw:.2e}")
    _check("loss finite", torch.isfinite(torch.tensor(val)).item(),
           f"loss={val:.3e}")


def test_cuda_if_available() -> None:
    """Same gradient-flow check on CUDA if a GPU is present."""
    print("\n[7] CUDA path (if available)")
    if not torch.cuda.is_available():
        _check("skipped (no CUDA)", True)
        return
    loss = FrequencyDecoupledDistillLoss(temperature=10.0, lf_size=8).cuda()
    s = torch.randn(2, 6, 64, 64, device="cuda", requires_grad=True)
    t = torch.randn(2, 6, 64, 64, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        val = loss(s, t)
    val.backward()
    _check("loss finite", torch.isfinite(val).item(), f"val={val.item():.3e}")
    _check("student grad finite",
           torch.isfinite(s.grad).all().item(), "grad has NaN/inf")


def main() -> int:
    print("=" * 60)
    print("FrequencyDecoupledDistillLoss sanity tests")
    print("=" * 60)
    tests = [
        test_dct_roundtrip,
        test_constant_input_dc_only,
        test_self_distill_zero,
        test_gradient_flow,
        test_cache_key,
        test_lf_size_clamp,
        test_cuda_if_available,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failures += 1
        except Exception as e:
            print(f"  ERROR in {t.__name__}: {type(e).__name__}: {e}")
            failures += 1
    print("\n" + "=" * 60)
    print(f"{'ALL PASSED' if failures == 0 else f'{failures} TEST(S) FAILED'}")
    print("=" * 60)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
