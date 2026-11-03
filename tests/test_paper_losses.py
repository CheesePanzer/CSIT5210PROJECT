"""独立的逐类别对 CE 参考实现，检查作者版 OVO 的值、梯度和 batch 行为。"""

import math
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from src.network.paper_losses import PaperLoss
from src.network.network_train_torch_manual import (
    PaperMLP, fit_and_select_lr, make_loader, predict, seed_everything, train_one_epoch,
)


def explicit_pair_reference(logits, labels):
    """按 i<j 独立建立两类 CE，每对仅使用本 batch 属于这两类的样本。"""
    k = logits.shape[1]
    terms = []
    for i in range(k):
        for j in range(i + 1, k):
            active = (labels == i) | (labels == j)
            if active.any():
                binary_labels = (labels[active] == j).long()
                terms.append(F.cross_entropy(logits[active][:, [i, j]], binary_labels))
    return torch.stack(terms).sum() / (k * (k - 1) / 2)


class AuthorLossTests(unittest.TestCase):
    def test_ovo_value_and_gradient_match_independent_pair_ce(self):
        for labels, k in (([0, 0, 0, 1, 2], 3), ([0, 0, 1], 4), ([2, 2, 2], 4)):
            seed_everything(23)
            logits = torch.randn(len(labels), k, dtype=torch.float64, requires_grad=True)
            reference_logits = logits.detach().clone().requires_grad_()
            labels = torch.tensor(labels)
            actual = PaperLoss("ovo", k)(logits, labels)
            expected = explicit_pair_reference(reference_logits, labels)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            actual.backward()
            expected.backward()
            torch.testing.assert_close(logits.grad, reference_logits.grad, rtol=1e-12, atol=1e-12)

    def test_zero_logits_scale_and_missing_pairs(self):
        labels = torch.tensor([0, 0, 0, 1, 2])
        actual = PaperLoss("ovo", 3)(torch.zeros(5, 3, dtype=torch.float64), labels)
        self.assertAlmostEqual(actual.item(), math.log(2), places=12)
        # 只有一个类别出现时，两对有效比较，分母仍为总共三对。
        actual = PaperLoss("ovo", 3)(torch.zeros(5, 3, dtype=torch.float64), torch.zeros(5, dtype=torch.long))
        self.assertAlmostEqual(actual.item(), 2 * math.log(2) / 3, places=12)

    def test_counts_are_local_to_batch_not_global_training_data(self):
        logits = torch.tensor([[1., 0., -1.], [0., 2., -1.], [0.5, -1., 1.]], dtype=torch.float64)
        labels = torch.tensor([0, 0, 1])
        criterion = PaperLoss("ovo", 3)
        actual = criterion(logits[:1], labels[:1])
        expected = explicit_pair_reference(logits[:1], labels[:1])
        torch.testing.assert_close(actual, expected)
        # 单个 batch 的值不要求等于“全量训练目标”的无偏分解，这正是旧版区别。
        self.assertFalse(torch.isclose(actual, criterion(logits, labels)))

    def test_ovr_is_mean_over_samples_and_classes(self):
        logits = torch.tensor([[0.2, -0.8, 1.1], [0.8, 0.4, -0.5]], dtype=torch.float64)
        labels = torch.tensor([0, 1])
        p = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
        expected = -sum(math.log(p[n, i] if i == label else 1 - p[n, i])
                        for n, label in enumerate(labels) for i in range(3)) / 6
        self.assertAlmostEqual(PaperLoss("ovr", 3)(logits, labels).item(), expected, places=12)

    def test_repeating_batch_preserves_loss_and_gradient(self):
        logits = torch.tensor([[1., 0., -1.], [0., 1., 0.5]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 2])
        criterion = PaperLoss("ovo", 3)
        original = criterion(logits, labels)
        repeated = criterion(logits.repeat(3, 1), labels.repeat(3))
        torch.testing.assert_close(original, repeated)
        torch.testing.assert_close(torch.autograd.grad(original, logits)[0],
                                   torch.autograd.grad(repeated, logits)[0])

    def test_extreme_logits_have_finite_loss_and_gradients(self):
        for name in ("ce", "ovr", "ovo"):
            logits = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
            loss = PaperLoss(name, 3)(logits, torch.tensor([1]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_ovo_double_precision_gradcheck(self):
        logits = torch.tensor([[0.2, -0.7, 1.1], [0.5, 0.9, -0.4]], dtype=torch.float64, requires_grad=True)
        labels = torch.tensor([0, 1])
        self.assertTrue(torch.autograd.gradcheck(lambda z: PaperLoss("ovo", 3)(z, labels), (logits,)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_agrees_with_cpu(self):
        logits = torch.tensor([[0.2, -0.7, 1.1], [0.5, 0.9, -0.4]])
        labels = torch.tensor([0, 1])
        for name in ("ce", "ovr", "ovo"):
            expected = PaperLoss(name, 3)(logits, labels)
            actual = PaperLoss(name, 3).cuda()(logits.cuda(), labels.cuda())
            torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)


class ManualTrainingTests(unittest.TestCase):
    def test_all_losses_learn_simple_data(self):
        torch.set_num_threads(1)
        X, y = np.tile(np.eye(3, dtype=np.float32), (8, 1)), np.tile(np.arange(3), 8)
        for name in ("ce", "ovr", "ovo"):
            seed_everything(42)
            model = PaperMLP(3, 3)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
            for _ in range(12):
                train_one_epoch(model, make_loader(X, y, 8, shuffle=True), PaperLoss(name, 3), optimizer, "cpu")
            np.testing.assert_array_equal(predict(model, make_loader(X, y, 8), "cpu"), y)

    def test_manual_early_stopping(self):
        X, y = np.eye(3, dtype=np.float32), np.arange(3)
        args = SimpleNamespace(batch_size=3, max_epochs=8, patience=2, learning_rates=[1e-12], device="cpu")
        with redirect_stdout(StringIO()):
            best, trials = fit_and_select_lr(X, y, X, y, 3, "ovo", "macro_f1", args, 42)
        self.assertEqual(best["best_epoch"], 1)
        self.assertEqual(len(trials[0]["history"]), 3)


if __name__ == "__main__":
    unittest.main()
