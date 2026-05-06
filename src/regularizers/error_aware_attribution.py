"""General error-aware attribution regularization for differentiable tabular models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class EAResult:
    ea_loss: torch.Tensor
    p_importance: torch.Tensor
    q_err: torch.Tensor
    diagnostics: dict
    ema_state: torch.Tensor


def _normalize(v: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    v = torch.clamp(v, min=eps)
    return v / (torch.sum(v) + eps)


def _gini(dist: torch.Tensor) -> torch.Tensor:
    x = _normalize(dist)
    sorted_x, _ = torch.sort(x)
    n = sorted_x.numel()
    idx = torch.arange(1, n + 1, device=sorted_x.device, dtype=sorted_x.dtype)
    return torch.sum((2.0 * idx - n - 1.0) * sorted_x) / (n + 1e-12)


def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - torch.mean(a)
    b = b - torch.mean(b)
    denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b)) + 1e-12
    return torch.sum(a * b) / denom


def gradient_importance(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    *,
    y_batch: torch.Tensor | None = None,
    task_type: str = "regression",
    score_type: str = "logprob",
    scalarize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Compute normalized gradient attribution p_theta for any differentiable model."""
    x = x_batch.detach().clone().requires_grad_(True)
    pred = model(x)
    if scalarize_fn is None:
        tt = str(task_type).lower()
        if tt == "classification":
            if y_batch is None:
                scalar = torch.max(pred, dim=1).values
            else:
                y = y_batch.detach().long()
                if y.ndim > 1:
                    y = torch.argmax(y, dim=1)
                y = y.view(-1)
                st = str(score_type).lower()
                if st == "margin":
                    zy = pred.gather(1, y.unsqueeze(1)).squeeze(1)
                    mask = torch.ones_like(pred, dtype=torch.bool)
                    mask.scatter_(1, y.unsqueeze(1), False)
                    z_others = pred.masked_fill(~mask, float("-inf"))
                    scalar = zy - torch.logsumexp(z_others, dim=1)
                elif st == "logit":
                    scalar = pred.gather(1, y.unsqueeze(1)).squeeze(1)
                else:  # logprob
                    logp = F.log_softmax(pred, dim=1)
                    scalar = logp.gather(1, y.unsqueeze(1)).squeeze(1)
        else:
            scalar = torch.mean(pred, dim=1)
    else:
        scalar = scalarize_fn(pred)
    g = torch.autograd.grad(torch.sum(scalar), x, create_graph=True, retain_graph=True)[0]
    p = torch.mean(torch.abs(g) * torch.abs(x), dim=0)
    return _normalize(p)


def error_importance(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    masking_mode: str = "permute",
    baseline_values: torch.Tensor | None = None,
    gate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ema_state: torch.Tensor | None = None,
    ema_beta: float = 0.9,
    prev_q: torch.Tensor | None = None,
    task_type: str = "regression",
    positive_clipping: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Compute normalized q_err from masked error deltas and EMA-smoothed target."""
    with torch.no_grad():
        x0 = x_batch.detach()
        y0 = y_batch.detach()
        tt = str(task_type).lower()
        if tt == "classification":
            y0 = y0.long()
            if y0.ndim > 1:
                y0 = torch.argmax(y0, dim=1)
            y0 = y0.view(-1)
        pred0 = model(gate_fn(x0) if gate_fn is not None else x0)
        if tt == "regression" and pred0.shape != y0.shape:
            md = min(pred0.shape[-1], y0.shape[-1])
            pred0 = pred0[..., :md]
            y0 = y0[..., :md]
        base_loss = float(loss_fn(pred0, y0).item())

        n_features = x0.shape[1]
        eta = torch.zeros(n_features, device=x0.device, dtype=x0.dtype)
        for j in range(n_features):
            xj = x0.clone()
            mode = masking_mode.lower()
            if mode == "permute":
                idx = torch.randperm(xj.shape[0], device=xj.device)
                xj[:, j] = xj[idx, j]
            elif mode == "noise":
                std = torch.std(xj[:, j])
                xj[:, j] = xj[:, j] + torch.randn_like(xj[:, j]) * (0.1 * std + 1e-8)
            else:  # mean
                if baseline_values is None:
                    xj[:, j] = torch.mean(xj[:, j])
                else:
                    xj[:, j] = float(baseline_values[j].item())
            predj = model(gate_fn(xj) if gate_fn is not None else xj)
            if tt == "regression" and predj.shape != y0.shape:
                md = min(predj.shape[-1], y0.shape[-1])
                predj = predj[..., :md]
                y_ref = y0[..., :md]
            else:
                y_ref = y0
            lj = float(loss_fn(predj, y_ref).item())
            delta = float(lj - base_loss)
            eta[j] = max(0.0, delta) if positive_clipping else delta

        if not positive_clipping:
            # Shift to non-negative before normalization to avoid invalid probabilities.
            eta = eta - torch.min(eta)

        q_raw = _normalize(eta)
        if ema_state is None:
            q_ema = q_raw.detach().clone()
        else:
            q_ema = ema_beta * ema_state + (1.0 - ema_beta) * q_raw.detach()
            q_ema = _normalize(q_ema)

        q_corr = float("nan") if prev_q is None else float(_corr(_normalize(q_raw), _normalize(prev_q)).item())
        q_safe = _normalize(q_raw)
        diagnostics = {
            "q_entropy": float((-q_safe * torch.log(q_safe)).sum().item()),
            "q_gini": float(_gini(q_safe).item()),
            "q_corr": q_corr,
            "eta_mean": float(torch.mean(eta).item()),
            "eta_std": float(torch.std(eta).item()),
            "masking_mode": masking_mode,
            "positive_clipping": bool(positive_clipping),
        }
        return q_raw, q_ema.detach(), diagnostics


def alignment_loss(
    p_importance: torch.Tensor,
    q_err: torch.Tensor,
    *,
    mode: str = "cosine_mse",
    alpha: float = 0.5,
) -> tuple[torch.Tensor, dict]:
    p = _normalize(p_importance)
    q = _normalize(q_err)
    mse = torch.mean((p - q) ** 2)
    cosine = 1.0 - F.cosine_similarity(p.unsqueeze(0), q.unsqueeze(0), dim=1, eps=1e-8).squeeze(0)

    m = mode.lower()
    if m == "mse":
        loss = mse
    elif m == "cosine":
        loss = cosine
    elif m == "js_mse":
        mvec = 0.5 * (p + q)
        js = 0.5 * (
            torch.sum(p * (torch.log(p) - torch.log(mvec)))
            + torch.sum(q * (torch.log(q) - torch.log(mvec)))
        )
        loss = alpha * js + (1.0 - alpha) * mse
    else:  # cosine_mse
        loss = alpha * cosine + (1.0 - alpha) * mse

    d = {
        "p_q_corr": float(_corr(p, q).detach().item()),
        "loss_mode": mode,
    }
    return loss, d


def compute_ea_regularizer(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    masking_mode: str = "permute",
    baseline_values: torch.Tensor | None = None,
    scalarize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    gate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ema_state: torch.Tensor | None = None,
    ema_beta: float = 0.9,
    prev_q: torch.Tensor | None = None,
    align_mode: str = "cosine_mse",
    align_alpha: float = 0.5,
    task_type: str = "regression",
    score_type: str = "logprob",
    positive_clipping: bool = True,
) -> EAResult:
    p = gradient_importance(
        model,
        x_batch,
        y_batch=y_batch,
        task_type=task_type,
        score_type=score_type,
        scalarize_fn=scalarize_fn,
    )
    q_raw, q_ema, q_diag = error_importance(
        model,
        x_batch,
        y_batch,
        loss_fn=loss_fn,
        masking_mode=masking_mode,
        baseline_values=baseline_values,
        gate_fn=gate_fn,
        ema_state=ema_state,
        ema_beta=ema_beta,
        prev_q=prev_q,
        task_type=task_type,
        positive_clipping=positive_clipping,
    )
    loss, align_diag = alignment_loss(p, q_ema, mode=align_mode, alpha=align_alpha)
    diag = {**q_diag, **align_diag}
    return EAResult(
        ea_loss=loss,
        p_importance=p,
        q_err=q_ema,
        diagnostics=diag,
        ema_state=q_ema.detach(),
    )


def _flatten_grads(grads: list[torch.Tensor | None]) -> torch.Tensor:
    parts = []
    for g in grads:
        if g is None:
            continue
        parts.append(g.reshape(-1))
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


def backward_with_eaar_projection(
    model: torch.nn.Module,
    main_loss: torch.Tensor,
    ea_loss: torch.Tensor,
    *,
    gamma: float,
    mode: str = "projected",
    eps: float = 1e-12,
) -> dict:
    """Set gradients for model params using task + EAAR with optional conflict projection."""
    params = [p for p in model.parameters() if p.requires_grad]
    g_main = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    g_ea = torch.autograd.grad(ea_loss, params, retain_graph=False, allow_unused=True)

    gm_flat = _flatten_grads(list(g_main))
    ge_flat = _flatten_grads(list(g_ea))
    denom = torch.sum(gm_flat * gm_flat) + eps
    dot = torch.sum(gm_flat * ge_flat)
    cosine = dot / (torch.sqrt(torch.sum(gm_flat * gm_flat) * torch.sum(ge_flat * ge_flat)) + eps)

    projected = False
    if str(mode).lower() == "projected" and float(dot.item()) < 0.0:
        projected = True
        scale = dot / denom
        ge_proj = []
        for gm, ge in zip(g_main, g_ea):
            if ge is None and gm is None:
                ge_proj.append(None)
            elif ge is None:
                ge_proj.append(torch.zeros_like(gm))
            elif gm is None:
                ge_proj.append(ge)
            else:
                ge_proj.append(ge - scale * gm)
        g_ea = tuple(ge_proj)

    for p, gm, ge in zip(params, g_main, g_ea):
        g = None
        if gm is not None:
            g = gm if g is None else (g + gm)
        if ge is not None:
            g = (gamma * ge) if g is None else (g + gamma * ge)
        p.grad = g

    return {
        "grad_conflict_dot": float(dot.detach().item()),
        "grad_conflict_cosine": float(cosine.detach().item()),
        "grad_projected": bool(projected),
        "combine_mode": str(mode).lower(),
    }
