import itertools
import unittest

import torch
from torch.nn.functional import one_hot

from adabmDCA.low_energy import _state_key, flood_fill_low_energy
from adabmDCA.statmech import compute_energy


def make_model(L: int = 4, q: int = 3) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    bias = torch.randn((L, q), generator=generator, dtype=torch.float64)
    coupling = torch.randn((L, q, L, q), generator=generator, dtype=torch.float64)
    coupling = 0.5 * (coupling + coupling.permute(2, 3, 0, 1))
    positions = torch.arange(L)
    coupling[positions, :, positions, :] = 0
    return {"bias": bias, "coupling_matrix": coupling}


def exact_component(seed, params, threshold):
    L, q = params["bias"].shape
    all_states = torch.tensor(list(itertools.product(range(q), repeat=L)))
    energies = compute_energy(one_hot(all_states, q).to(params["bias"].dtype), params)
    allowed = {tuple(state.tolist()) for state, energy in zip(all_states, energies) if energy <= threshold}
    reached = {tuple(seed.tolist())}
    frontier = [tuple(seed.tolist())]
    while frontier:
        state = frontier.pop()
        for site in range(L):
            for value in range(q):
                child = state[:site] + (value,) + state[site + 1 :]
                if child in allowed and child not in reached:
                    reached.add(child)
                    frontier.append(child)
    return reached


class LowEnergyFloodFillTests(unittest.TestCase):
    def test_state_keys_are_exact_compact_bytes(self):
        state = torch.tensor([0, 1, 20, 255], dtype=torch.int64)
        key = _state_key(state, q=256)

        self.assertIsInstance(key, bytes)
        self.assertEqual(key, bytes([0, 1, 20, 255]))
        self.assertEqual(len(key), len(state))
        self.assertNotEqual(key, _state_key(torch.tensor([0, 1, 20, 254]), q=256))

    def test_state_keys_support_large_custom_alphabets(self):
        state = torch.tensor([0, 256, 511], dtype=torch.int64)

        self.assertEqual(_state_key(state, q=512), state.numpy().tobytes())

    def test_matches_exact_enumeration_and_energies(self):
        params = make_model()
        seed = torch.tensor([0, 1, 2, 0])
        seed_energy = compute_energy(one_hot(seed[None], 3).to(torch.float64), params)[0]
        threshold = float(seed_energy + 0.75)

        result = flood_fill_low_energy(seed[None], params, threshold, max_states=None, max_frontier=None)
        expected = exact_component(seed, params, threshold)

        self.assertTrue(result.complete)
        self.assertEqual(result.termination_reason, "frontier_exhausted")
        self.assertEqual({tuple(state.tolist()) for state in result.sequences}, expected)
        exact_energies = compute_energy(result.to_one_hot(3, torch.float64), params)
        torch.testing.assert_close(result.energies, exact_energies, rtol=1e-10, atol=1e-10)

    def test_deduplicates_seeds_in_the_same_component(self):
        params = {
            "bias": torch.zeros((3, 2), dtype=torch.float64),
            "coupling_matrix": torch.zeros((3, 2, 3, 2), dtype=torch.float64),
        }
        seeds = torch.tensor([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        result = flood_fill_low_energy(seeds, params, 0.0, max_states=None, max_frontier=None)

        self.assertTrue(result.complete)
        self.assertEqual(len(result.sequences), 2**3)
        self.assertEqual(result.seed_component_ids.tolist(), [0, 0, 0])
        self.assertEqual(result.component_ids.unique().tolist(), [0])

    def test_skips_seed_above_threshold(self):
        params = {
            "bias": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "coupling_matrix": torch.zeros((2, 2, 2, 2)),
        }
        seeds = torch.tensor([[0, 0], [1, 1]])
        result = flood_fill_low_energy(seeds, params, -1.5, max_states=None, max_frontier=None)

        self.assertEqual(result.eligible_seed_mask.tolist(), [True, False])
        self.assertEqual(result.seed_component_ids.tolist(), [0, -1])
        self.assertEqual({tuple(state.tolist()) for state in result.sequences}, {(0, 0)})

    def test_distinguishes_disconnected_components(self):
        coupling = torch.zeros((2, 2, 2, 2), dtype=torch.float64)
        coupling[0, :, 1, :] = 2.0 * torch.eye(2, dtype=torch.float64)
        coupling[1, :, 0, :] = 2.0 * torch.eye(2, dtype=torch.float64)
        params = {
            "bias": torch.zeros((2, 2), dtype=torch.float64),
            "coupling_matrix": coupling,
        }
        seeds = torch.tensor([[0, 0], [1, 1]])
        result = flood_fill_low_energy(seeds, params, -1.0, max_states=None, max_frontier=None)

        self.assertTrue(result.complete)
        self.assertEqual({tuple(state.tolist()) for state in result.sequences}, {(0, 0), (1, 1)})
        self.assertEqual(result.seed_component_ids.tolist(), [0, 1])
        self.assertEqual(result.component_ids.tolist(), [0, 1])

    def test_exclusive_threshold_omits_equal_energy_states(self):
        params = {
            "bias": torch.zeros((2, 2)),
            "coupling_matrix": torch.zeros((2, 2, 2, 2)),
        }
        seed = one_hot(torch.tensor([[0, 0]]), 2).float()
        result = flood_fill_low_energy(
            seed,
            params,
            0.0,
            inclusive=False,
            max_states=None,
            max_frontier=None,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.sequences.shape, (0, 2))
        self.assertEqual(result.seed_component_ids.tolist(), [-1])

    def test_reports_truncation_at_state_limit(self):
        params = {
            "bias": torch.zeros((4, 2)),
            "coupling_matrix": torch.zeros((4, 2, 4, 2)),
        }
        result = flood_fill_low_energy(torch.zeros((1, 4), dtype=torch.int64), params, 0.0, max_states=5)

        self.assertFalse(result.complete)
        self.assertEqual(result.termination_reason, "max_states")
        self.assertEqual(len(result.sequences), 5)

    def test_limits_gap_characters_in_queued_states(self):
        params = {
            "bias": torch.zeros((3, 2), dtype=torch.float64),
            "coupling_matrix": torch.zeros((3, 2, 3, 2), dtype=torch.float64),
        }
        seeds = torch.tensor([[1, 1, 1], [0, 0, 1]])
        result = flood_fill_low_energy(
            seeds,
            params,
            0.0,
            max_states=None,
            max_frontier=None,
            max_gaps=1,
            gap_token=0,
        )

        self.assertTrue(result.complete)
        self.assertEqual(result.eligible_seed_mask.tolist(), [True, False])
        self.assertEqual(result.seed_component_ids.tolist(), [0, -1])
        self.assertEqual(len(result.sequences), 4)
        self.assertTrue(torch.all((result.sequences == 0).sum(dim=1) <= 1))

    def test_rejects_invalid_gap_constraint(self):
        params = {
            "bias": torch.zeros((2, 2)),
            "coupling_matrix": torch.zeros((2, 2, 2, 2)),
        }
        seeds = torch.zeros((1, 2), dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "max_gaps"):
            flood_fill_low_energy(seeds, params, 0.0, max_gaps=-1)
        with self.assertRaisesRegex(ValueError, "gap_token"):
            flood_fill_low_energy(seeds, params, 0.0, gap_token=2)

    def test_progress_callback_reports_intervals_and_final_state(self):
        params = {
            "bias": torch.zeros((3, 2)),
            "coupling_matrix": torch.zeros((3, 2, 3, 2)),
        }
        seeds = torch.tensor([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
        updates = []
        result = flood_fill_low_energy(
            seeds,
            params,
            0.0,
            max_states=None,
            max_frontier=100,
            progress=updates.append,
            progress_interval=5,
        )

        self.assertGreaterEqual(len(updates), 2)
        self.assertTrue(all(update.tested >= 5 for update in updates[:-1]))
        self.assertTrue(all(a.tested < b.tested for a, b in zip(updates[:-2], updates[1:-1])))
        final = updates[-1]
        self.assertEqual(final.tested, result.num_tested)
        self.assertEqual(final.accepted, len(result.sequences))
        self.assertEqual(final.frontier, 0)
        self.assertEqual(final.unresolved_seeds, 0)
        self.assertEqual(final.eligible_seeds, 2)
        self.assertTrue(final.complete)
        self.assertEqual(final.termination_reason, "frontier_exhausted")

    def test_rejects_invalid_progress_interval(self):
        params = {
            "bias": torch.zeros((2, 2)),
            "coupling_matrix": torch.zeros((2, 2, 2, 2)),
        }
        with self.assertRaisesRegex(ValueError, "progress_interval"):
            flood_fill_low_energy(torch.zeros((1, 2), dtype=torch.int64), params, 0.0, progress_interval=0)

    def test_rejects_non_symmetric_couplings(self):
        params = make_model()
        params["coupling_matrix"][0, 0, 1, 0] += 1.0
        with self.assertRaisesRegex(ValueError, "symmetric"):
            flood_fill_low_energy(torch.zeros((1, 4), dtype=torch.int64), params, 0.0)


if __name__ == "__main__":
    unittest.main()
