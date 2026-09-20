"""Unit tests for distance-weighted ordinal cross-entropy loss."""

import torch

from jah.training.adapter import compute_decision_loss


def test_cross_entropy_equivalence():
    # Label token IDs map positions 0, 1, 2
    label_token_ids = [10, 20, 30]
    # vocab size at least 31
    full_logits = torch.zeros(1, 35, requires_grad=True)
    with torch.no_grad():
        full_logits[0, 10] = 2.0
        full_logits[0, 20] = 1.0
        full_logits[0, 30] = -1.0

    target = [0]
    loss_ce = compute_decision_loss(full_logits, target, label_token_ids, loss_type="cross_entropy")
    assert loss_ce.item() > 0


def test_ordinal_penalty_increases_with_distance():
    label_token_ids = [10, 20, 30, 40, 50]  # 5 classes (0..4)
    target = [0]  # Target is class 0

    # Model A predicts class 1 (off by 1)
    logits_near = torch.zeros(1, 60)
    logits_near[0, 20] = 10.0  # class 1

    # Model B predicts class 4 (off by 4)
    logits_far = torch.zeros(1, 60)
    logits_far[0, 50] = 10.0  # class 4

    # With standard CE, both have near-identical CE loss because neither predicted class 0
    loss_ce_near = compute_decision_loss(logits_near, target, label_token_ids, loss_type="cross_entropy")
    loss_ce_far = compute_decision_loss(logits_far, target, label_token_ids, loss_type="cross_entropy")
    assert abs(loss_ce_near.item() - loss_ce_far.item()) < 1e-3

    # With ordinal loss, predicting class 4 incurs significantly higher penalty than class 1
    loss_ord_near = compute_decision_loss(logits_near, target, label_token_ids, loss_type="ordinal")
    loss_ord_far = compute_decision_loss(logits_far, target, label_token_ids, loss_type="ordinal")
    assert loss_ord_far.item() > loss_ord_near.item() + 0.5


def test_ordinal_loss_zero_distance_when_perfect():
    label_token_ids = [10, 20, 30]
    target = [1]  # Target is class 1

    # Confident correct prediction
    logits_correct = torch.zeros(1, 40)
    logits_correct[0, 20] = 20.0

    loss_ce = compute_decision_loss(logits_correct, target, label_token_ids, loss_type="cross_entropy")
    loss_ord = compute_decision_loss(logits_correct, target, label_token_ids, loss_type="ordinal")
    # Distance penalty for class 1 to class 1 is 0.0
    assert abs(loss_ord.item() - loss_ce.item()) < 1e-4


def test_ordinal_loss_backward_gradient():
    label_token_ids = [10, 20, 30, 40]
    logits = torch.randn(2, 50, requires_grad=True)
    targets = [0, 3]

    loss = compute_decision_loss(logits, targets, label_token_ids, loss_type="ordinal", ordinal_penalty_weight=2.0)
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad[:, label_token_ids] != 0.0)


def test_ordinal_loss_with_soft_targets():
    label_token_ids = [10, 20, 30]
    logits = torch.randn(1, 40, requires_grad=True)
    soft_targets = [[0.1, 0.8, 0.1]]

    loss = compute_decision_loss(logits, soft_targets, label_token_ids, loss_type="ordinal")
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None
