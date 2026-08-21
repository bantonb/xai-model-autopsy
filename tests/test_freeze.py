"""Assert the ConvNeXt-Small backbone is genuinely frozen.

The thesis's trainer set .eval() but never .requires_grad_(False), so
backbone weights were fine-tuned end-to-end despite the thesis text claiming
a frozen backbone. This test enforces both.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pipeline.embed import FEATURE_DIM, build_frozen_backbone


def test_no_backbone_parameter_receives_gradient():
    backbone = build_frozen_backbone(device="cpu")
    head = nn.Linear(FEATURE_DIM, 2)
    x = torch.randn(2, 3, 224, 224)

    # Realistic training-step shape: features do NOT go through no_grad here;
    # if requires_grad_(False) worked, backbone params still have grad=None.
    feats = backbone(x)
    if feats.ndim > 2:
        feats = feats.mean(dim=list(range(2, feats.ndim)))
    logits = head(feats)
    loss = logits.sum()
    loss.backward()

    grad_present = [
        n for n, p in backbone.named_parameters() if p.grad is not None
    ]
    assert not grad_present, (
        f"{len(grad_present)} backbone parameters accumulated gradients: "
        f"first = {grad_present[0]}"
    )

    grad_absent_head = [n for n, p in head.named_parameters() if p.grad is None]
    assert not grad_absent_head, (
        f"head parameters missing gradient (should be present): {grad_absent_head}"
    )


def test_backbone_is_in_eval_mode():
    backbone = build_frozen_backbone(device="cpu")
    assert not backbone.training, "backbone .training is True — should be .eval()"


def test_backbone_output_dim_matches_declared():
    backbone = build_frozen_backbone(device="cpu")
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        y = backbone(x)
    if y.ndim > 2:
        y = y.mean(dim=list(range(2, y.ndim)))
    assert y.shape == (1, FEATURE_DIM), (
        f"backbone output shape {tuple(y.shape)} != (1, {FEATURE_DIM})"
    )
