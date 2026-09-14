"""Frozen-checkpoint mode and normalization interventions; never a trained policy."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from itertools import product

import torch
from torch import nn


@contextmanager
def mode(
    model, *, encoder_batch=False, predictor_batch=False, dropout=False, seed=2026
):
    """Restore buffers, modes and RNG even when a diagnostic raises."""
    modules = list(model.modules())
    flags = [m.training for m in modules]
    buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
    device = next(model.parameters()).device
    devices = [device.index] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            model.eval()
            model.predictor.train(dropout)
            model.projector.train(encoder_batch)
            model.pred_projector.train(predictor_batch)
            yield
    finally:
        with torch.no_grad():
            for name, value in model.named_buffers():
                value.copy_(buffers[name])
        for module, flag in zip(modules, flags, strict=True):
            module.training = flag


def _project(model, cls):
    return model._apply_projector(model.projector, cls)


def _score(model, z, actions):
    device = z.device
    devices = [device.index] if device.type == "cuda" else []
    # Pair dropout masks so action sensitivity does not include mask noise.
    with torch.random.fork_rng(devices=devices):
        predicted = model.predict(z[:, :-1], actions)
    wrong = model.predict(z[:, :-1], (actions + 1) % 9)
    target = z[:, 1:]
    errors = (predicted - target).square().mean((0, 2))
    persistence = (z[:, :-1] - target).square().mean((0, 2))
    result = {
        "prediction_mse_by_position": errors.tolist(),
        "persistence_mse_by_position": persistence.tolist(),
        "all_positions_mse": errors.mean().item(),
        "last_prediction_mse": errors[-1].item(),
        "last_persistence_mse": persistence[-1].item(),
        "last_ratio": (errors[-1] / persistence[-1]).item()
        if persistence[-1] > 0
        else None,
        "wrong_action_last_mse": (wrong[:, -1] - target[:, -1]).square().mean().item(),
        "action_sensitivity_mse": (wrong - predicted).square().mean().item(),
        "target_std": target.flatten(0, 1).std(0, unbiased=False).mean().item(),
        "prediction_std": predicted.flatten(0, 1).std(0, unbiased=False).mean().item(),
    }
    if not all(torch.isfinite(v).all() for v in (predicted, target, wrong)):
        raise RuntimeError("nonfinite normalization diagnostic")
    return result


def _bn(module):
    layers = [m for m in module.modules() if isinstance(m, nn.BatchNorm1d)]
    if len(layers) != 1:
        raise ValueError("audit expects one BatchNorm per projector")
    return layers[0]


def capture_input(module, operation):
    captured = []
    handle = module.register_forward_pre_hook(
        lambda _m, args: captured.append(args[0].detach().clone())
    )
    try:
        operation()
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError("expected exactly one normalization input")
    return captured[0]


def replace_moments(bn, values):
    """Exact train-population moments, distinct from EMA running estimates."""
    with torch.no_grad():
        bn.running_mean.copy_(values.mean(0))
        bn.running_var.copy_(values.var(0, unbiased=False))


def moment_report(bn, values):
    variance = values.var(0, unbiased=False)
    relative = variance / (bn.running_var + bn.eps)
    return {
        "observations": values.shape[0],
        "num_batches_tracked": bn.num_batches_tracked.item(),
        "mean_shift_in_running_sd": (
            (values.mean(0) - bn.running_mean) / (bn.running_var + bn.eps).sqrt()
        )
        .abs()
        .mean()
        .item(),
        "batch_to_running_variance_median": relative.median().item(),
        "batch_to_running_variance_min": relative.min().item(),
        "batch_to_running_variance_max": relative.max().item(),
    }


def audit(
    model,
    train_pixels,
    train_actions,
    validation_pixels,
    validation_actions,
    matched_indices,
):
    """Factorial mode audit plus train-only buffer interventions on a copy.

    Cached CLS features are exact here: reference encoder dropout is zero.
    No optimizer, gradients, native collection, or checkpoint write occurs.
    """
    if model.config.encoder_dropout or model.config.encoder_attention_dropout:
        raise ValueError("CLS caching requires deterministic encoder")
    original = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    with mode(model):
        train_cls, _ = model._encode_tokens(train_pixels)
        val_cls, _ = model._encode_tokens(validation_pixels)
    groups = {
        "matched_train": (train_cls[matched_indices], train_actions[matched_indices]),
        "validation": (val_cls, validation_actions),
    }
    result = {
        "modes": {},
        "moments": {},
        "future_substitution": {},
        "calibration": {},
        "optimizer_updates": 0,
        "matched_indices": matched_indices.tolist(),
        "calibration_semantics": (
            "train-window-weighted population moments; disposable copies only"
        ),
    }
    for enc, pred, drop in product((False, True), repeat=3):
        key = f"encoder_batch={enc},predictor_batch={pred},dropout={drop}"
        result["modes"][key] = {}
        for name, (cls, actions) in groups.items():
            scores = []
            for seed in (2026, 2027, 2028) if drop else (2026,):
                with mode(
                    model,
                    encoder_batch=enc,
                    predictor_batch=pred,
                    dropout=drop,
                    seed=seed,
                ):
                    scores.append(_score(model, _project(model, cls), actions))
            result["modes"][key][name] = scores
    for enc in (False, True):
        cls = train_cls[matched_indices]
        changed = cls.clone()
        changed[:, -1] = cls[:, 0]  # context identical; substitute future only
        with mode(model, encoder_batch=enc):
            z = _project(model, cls)
        with mode(model, encoder_batch=enc):
            alternate = _project(model, changed)
        result["future_substitution"][str(enc)] = {
            "context_latent_mse": (z[:, :-1] - alternate[:, :-1])
            .square()
            .mean()
            .item(),
            "context_max_abs": (z[:, :-1] - alternate[:, :-1]).abs().max().item(),
        }
    with mode(model):
        encoder_input = capture_input(
            _bn(model.projector), lambda: _project(model, train_cls)
        )
        z = _project(model, train_cls)
        predictor_input = capture_input(
            _bn(model.pred_projector), lambda: model.predict(z[:, :-1], train_actions)
        )
        result["moments"] = {
            "encoder": moment_report(_bn(model.projector), encoder_input),
            "predictor_under_eval_encoder": moment_report(
                _bn(model.pred_projector), predictor_input
            ),
        }
    for enc, pred in ((False, False), (False, True), (True, False), (True, True)):
        candidate = copy.deepcopy(model).eval().requires_grad_(False)
        with torch.no_grad():
            if enc:
                replace_moments(_bn(candidate.projector), encoder_input)
            if pred:
                cz = _project(candidate, train_cls)
                values = capture_input(
                    _bn(candidate.pred_projector),
                    lambda candidate=candidate, cz=cz: candidate.predict(
                        cz[:, :-1], train_actions
                    ),
                )
                replace_moments(_bn(candidate.pred_projector), values)
            key = f"encoder_recalibrated={enc},predictor_recalibrated={pred}"
            result["calibration"][key] = {
                name: _score(candidate, _project(candidate, cls), actions)
                for name, (cls, actions) in {
                    "train": (train_cls, train_actions),
                    "validation": (val_cls, validation_actions),
                }.items()
            }
        del candidate
    for key, value in model.state_dict().items():
        if not torch.equal(original[key], value.detach().cpu()):
            raise RuntimeError(f"audit mutated original model: {key}")
    result["original_state_unchanged"] = True
    return result
