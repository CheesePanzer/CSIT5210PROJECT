"""CE, OVR, and OVO losses for the paper's mini-batch training setup."""

import torch
from torch import nn
from torch.nn import functional as F


class PaperLoss(nn.Module):
    """CE, OVR, and OVO losses based on the paper's definitions."""

    def __init__(self, name, num_classes, class_counts):
        super().__init__()
        if name not in ("ce", "ovr", "ovo"):
            raise ValueError(f"Unknown loss: {name}")
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("num_classes must be an integer >= 2.")
        self.name = name
        self.num_classes = num_classes
        self.num_pairs = num_classes * (num_classes - 1) // 2
        class_counts = torch.as_tensor(class_counts, dtype=torch.float64)
        if class_counts.shape != (num_classes,) or torch.any(class_counts <= 0):
            raise ValueError("class_counts must contain one positive count per class.")
        self.register_buffer("class_counts", class_counts)

    def forward(self, logits, targets):
        self._validate_inputs(logits, targets)

        if self.name == "ce":
            return F.cross_entropy(logits, targets)
        if self.name == "ovr":
            return self._ovr_loss(logits, targets)
        return self._ovo_loss(logits, targets)

    def _validate_inputs(self, logits, targets):
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError("logits must have shape [batch_size, num_classes].")
        if targets.ndim != 1 or len(targets) != len(logits) or len(targets) == 0:
            raise ValueError("targets must be a nonempty vector matching the batch size.")

    def _ovr_loss(self, logits, targets):
        """Compute the OVR objective corresponding to paper Eq. (9).

        ``logits[n, i]`` is f_i(x_n), ``targets[n]`` identifies O_i, and
        ``probabilities[n, i]`` is p_i(x_n) from paper Eq. (7). The BCE
        reduction averages the B*K terms, matching the 1/N and sum-over-i
        structure of Eq. (9) for the current batch.
        """
        # targets_one_hot[n, i] is 1{y_n=i} in paper Eq. (9).
        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes,
        ).to(dtype=logits.dtype)

        # probabilities[n, i] = p_i(x_n) = exp(f_i) / sum_j exp(f_j),
        # which is exactly paper Eq. (7).
        probabilities = F.softmax(logits, dim=1)

        # BCE expands to:
        # -mean_n,i [1{y_n=i} log p_i(x_n)
        #            + (1-1{y_n=i}) log(1-p_i(x_n))],
        # i.e. paper Eq. (9), with the mean taken over the current batch.
        return F.binary_cross_entropy(probabilities, targets_one_hot)

    def _ovo_loss(self, logits, targets):
        """Compute the OVO objective using the paper's pairwise terms.

        Paper notation mapped to this function:

        - B = current batch size;
        - K = ``self.num_classes``;
        - f_i(x_n) = ``logits[n, i]``;
        - O_i = samples whose target is class i;
        - p_i,j(x_n) = exp(f_i)/(exp(f_i)+exp(f_j)), paper Eq. (10).

        The pairwise probability follows Eq. (10), while the class-pair
        denominator uses fixed counts from the complete training partition.
        Training still uses mini-batches of size 8.
        """
        # targets_one_hot[n, i] is 1{y_n=i}; it later removes the invalid
        # self-comparison i=j from the K-way comparison matrix.
        targets_one_hot = F.one_hot(
            targets,
            num_classes=self.num_classes,
        ).bool()

        # counts[i] = |O_i| from the complete training partition, not the
        # number of class-i samples in this mini-batch.
        counts = self.class_counts.to(device=logits.device, dtype=logits.dtype)

        # true_class_logits[n, 0] = f_{y_n}(x_n).
        true_class_logits = logits.gather(1, targets[:, None])

        # pairwise_logits[n, j] = f_{y_n}(x_n) - f_j(x_n).
        # sigmoid(pairwise_logits) equals p_{y_n,j}(x_n) in Eq. (10).
        pairwise_logits = true_class_logits - logits

        # pairwise_loss[n, j] = -log p_{y_n,j}(x_n), the negative
        # log-likelihood term in Eqs. (11) and (12).
        pairwise_loss = -F.logsigmoid(pairwise_logits)

        # pair_counts[n, j] = |O_{y_n}| + |O_j| from Eq. (11)/(12).
        pair_counts = counts[targets, None] + counts[None, :]

        # Apply the 1/(|O_i|+|O_j|) pairwise weight from Eqs. (11)-(13).
        weighted_loss = pairwise_loss / pair_counts

        # Remove j=y_n because a class must not be compared with itself.
        weighted_loss = weighted_loss.masked_fill(targets_one_hot, 0.0)

        # Keep batch size=8; this is the stochastic batch estimate of the
        # paper objective with fixed full-training class counts.
        return 2.0 * weighted_loss.sum() / len(targets)
