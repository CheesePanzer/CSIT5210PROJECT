"""调包组合后的公式等价性，以及 skorch 训练/早停的集成检查。"""

import unittest
from types import SimpleNamespace

try:
    import skorch
except ModuleNotFoundError:
    raise unittest.SkipTest("Install requirements_skorch.txt, or run with the project .venv.")

import numpy as np
import torch

from src.network.network_train_pytorch import PaperLoss
from src.network.network_train_skorch import CombinedPaperLoss, build_classifier, seed_everything


class SkorchVersionTests(unittest.TestCase):
    def test_builtin_loss_combinations_match_paper_value_and_gradient(self):
        logits = torch.tensor(
            [[0.2, -0.8, 1.1, 0.3], [0.8, 0.4, -0.5, 1.2], [-0.3, 0.6, 0.4, 1.1]],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 0, 1])
        # 完整训练分区包含类别 2/3；当前 batch 没有，仍应保留其比较和计数。
        for name in ("ce", "ovr", "ovo"):
            expected_logits = logits.clone().requires_grad_()
            actual_logits = logits.clone().requires_grad_()
            expected = PaperLoss(name, [10, 3, 2, 1])(expected_logits, labels)
            actual = CombinedPaperLoss(name, [10, 3, 2, 1])(actual_logits, labels)
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
            expected.backward()
            actual.backward()
            torch.testing.assert_close(actual_logits.grad, expected_logits.grad, rtol=1e-11, atol=1e-11)

    def test_extreme_logits_remain_finite_and_shift_invariant(self):
        logits = torch.tensor([[1000.0, -1000.0, 0.0]], dtype=torch.float64)
        labels = torch.tensor([1])
        for name in ("ce", "ovr", "ovo"):
            criterion = CombinedPaperLoss(name, [1, 1, 0])
            values = logits.clone().requires_grad_()
            loss = criterion(values, labels)
            shifted = criterion(logits + 100, labels)
            torch.testing.assert_close(loss, shifted)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertTrue(torch.isfinite(values.grad).all().item())

    def test_skorch_fit_predict_learns_all_three_losses(self):
        torch.set_num_threads(1)
        X = np.tile(np.eye(3, dtype=np.float32), (8, 1))
        y = np.tile(np.arange(3, dtype=np.int64), 8)
        args = SimpleNamespace(batch_size=8, max_epochs=12, patience=2, device="cpu")
        for name in ("ce", "ovr", "ovo"):
            seed_everything(42)
            classifier = build_classifier(3, 3, name, [8, 8, 8], 0.005, args)
            classifier.fit(X, y)
            np.testing.assert_array_equal(classifier.predict(X), y)
            np.testing.assert_allclose(classifier.predict_proba(X).sum(axis=1), 1, atol=1e-6)

    def test_library_early_stopping_uses_fixed_validation_partition(self):
        torch.set_num_threads(1)
        seed_everything(42)
        X = np.eye(3, dtype=np.float32)
        y = np.arange(3, dtype=np.int64)
        args = SimpleNamespace(batch_size=3, max_epochs=8, patience=2, device="cpu")
        classifier = build_classifier(
            3, 3, "ce", [1, 1, 1], 1e-12, args,
            validation=(X, y), criterion="macro_f1",
        )
        classifier.fit(X, y)
        self.assertEqual(len(classifier.history), 3)
        stopper = dict(classifier.callbacks_)["paper_early_stop"]
        self.assertEqual(stopper.best_epoch_, 1)


if __name__ == "__main__":
    unittest.main()
