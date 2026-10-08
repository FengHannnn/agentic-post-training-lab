import math
import unittest

import torch

from agentic_post_training.algorithms import (
    dpo_loss,
    group_relative_advantages,
    grpo_clipped_loss,
    masked_cross_entropy,
    sequence_log_probs,
)


class CausalLossTests(unittest.TestCase):
    def test_masked_cross_entropy_ignores_prompt_tokens(self) -> None:
        labels = torch.tensor([[0, 1, 0, 1]])
        mask = torch.tensor([[0, 0, 1, 1]])
        logits = torch.zeros(1, 4, 2)

        baseline = masked_cross_entropy(logits, labels, mask)
        logits[:, 0, :] = torch.tensor([100.0, -100.0])
        changed_prompt = masked_cross_entropy(logits, labels, mask)

        self.assertTrue(torch.allclose(baseline, changed_prompt))
        self.assertTrue(torch.allclose(baseline, torch.tensor(math.log(2.0))))

    def test_sequence_log_probs_sum_and_average(self) -> None:
        labels = torch.tensor([[0, 1, 0]])
        mask = torch.tensor([[0, 1, 1]])
        logits = torch.zeros(1, 3, 2)

        total = sequence_log_probs(logits, labels, mask)
        average = sequence_log_probs(logits, labels, mask, average=True)

        self.assertTrue(torch.allclose(total, torch.tensor([-2.0 * math.log(2.0)])))
        self.assertTrue(torch.allclose(average, torch.tensor([-math.log(2.0)])))

    def test_empty_response_mask_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            masked_cross_entropy(
                torch.zeros(1, 3, 2),
                torch.tensor([[0, 1, 0]]),
                torch.zeros(1, 3),
            )


class DPOTests(unittest.TestCase):
    def test_preferred_policy_has_lower_loss_and_perfect_accuracy(self) -> None:
        good = dpo_loss(
            torch.tensor([-1.0]),
            torch.tensor([-3.0]),
            torch.tensor([-2.0]),
            torch.tensor([-2.0]),
            beta=1.0,
        )
        neutral = dpo_loss(
            torch.tensor([-2.0]),
            torch.tensor([-2.0]),
            torch.tensor([-2.0]),
            torch.tensor([-2.0]),
            beta=1.0,
        )

        self.assertLess(good.loss.item(), neutral.loss.item())
        self.assertEqual(good.preference_accuracy.item(), 1.0)
        self.assertGreater(good.chosen_rewards.item(), good.rejected_rewards.item())

    def test_dpo_backpropagates_only_through_policy(self) -> None:
        policy_chosen = torch.tensor([-2.0], requires_grad=True)
        policy_rejected = torch.tensor([-2.0], requires_grad=True)
        output = dpo_loss(
            policy_chosen,
            policy_rejected,
            torch.tensor([-2.0]),
            torch.tensor([-2.0]),
            beta=0.5,
        )
        output.loss.backward()

        self.assertLess(policy_chosen.grad.item(), 0.0)
        self.assertGreater(policy_rejected.grad.item(), 0.0)


class GRPOTests(unittest.TestCase):
    def test_group_advantages_are_normalized(self) -> None:
        rewards = torch.tensor([[1.0, 0.4, 0.0, -0.2]])
        advantages = group_relative_advantages(rewards)

        self.assertTrue(torch.allclose(advantages.mean(dim=1), torch.zeros(1), atol=1e-6))
        self.assertTrue(
            torch.allclose(
                advantages.std(dim=1, unbiased=False),
                torch.ones(1),
                atol=1e-6,
            )
        )
        self.assertGreater(advantages[0, 0].item(), 0.0)
        self.assertLess(advantages[0, -1].item(), 0.0)

    def test_equal_rewards_produce_no_learning_signal(self) -> None:
        advantages = group_relative_advantages(torch.ones(2, 4))
        self.assertTrue(torch.equal(advantages, torch.zeros_like(advantages)))

    def test_positive_advantage_is_clipped(self) -> None:
        new = torch.tensor([[[math.log(2.0)]]], requires_grad=True)
        old = torch.zeros_like(new)
        mask = torch.ones_like(new)
        result = grpo_clipped_loss(
            new,
            old,
            torch.tensor([[1.0]]),
            mask,
            clip_epsilon=0.2,
        )

        self.assertAlmostEqual(result.loss.item(), -1.2, places=6)
        self.assertEqual(result.clip_fraction.item(), 1.0)

    def test_response_mask_excludes_padding(self) -> None:
        new = torch.zeros(1, 1, 2)
        old = torch.zeros_like(new)
        mask = torch.tensor([[[1.0, 0.0]]])
        advantages = torch.tensor([[1.0]])

        baseline = grpo_clipped_loss(new, old, advantages, mask)
        new[:, :, 1] = 100.0
        padded_token_changed = grpo_clipped_loss(new, old, advantages, mask)

        self.assertTrue(torch.allclose(baseline.loss, padded_token_changed.loss))

    def test_kl_estimator_is_non_negative(self) -> None:
        new = torch.tensor([[[-1.0, -2.0]]])
        old = new.clone()
        reference = torch.tensor([[[-1.5, -1.5]]])
        result = grpo_clipped_loss(
            new,
            old,
            torch.tensor([[0.0]]),
            torch.ones_like(new),
            reference_log_probs=reference,
            beta=0.1,
        )
        self.assertGreaterEqual(result.mean_kl.item(), 0.0)
        self.assertGreaterEqual(result.loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

