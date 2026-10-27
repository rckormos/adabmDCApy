import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from adabmDCA.replica_exchange import (
    _adapt_beta_ladder,
    _attempt_adjacent_swaps,
    geometric_beta_ladder,
    replica_exchange_sampling,
)
from adabmDCA.io import load_chains, save_params
from adabmDCA.statmech import compute_energy


def zero_model(L=3, q=2, dtype=torch.float64):
    return {
        "bias": torch.zeros((L, q), dtype=dtype),
        "coupling_matrix": torch.zeros((L, q, L, q), dtype=dtype),
    }


class ReplicaExchangeTests(unittest.TestCase):
    def test_geometric_ladder_is_descending_with_requested_endpoints(self):
        betas = geometric_beta_ladder(5, beta_min=0.1, beta_max=1.0, dtype=torch.float64)

        self.assertEqual(len(betas), 5)
        self.assertTrue(torch.all(betas[:-1] > betas[1:]))
        torch.testing.assert_close(betas[[0, -1]], torch.tensor([1.0, 0.1], dtype=torch.float64))

    def test_zero_energy_adjacent_swaps_are_always_accepted(self):
        params = zero_model(L=1)
        replicas = torch.zeros((2, 4, 1, 2), dtype=torch.float64)
        replicas[0, :, 0, 0] = 1
        replicas[1, :, 0, 1] = 1
        attempts = torch.zeros(1, dtype=torch.int64)
        accepts = torch.zeros(1, dtype=torch.int64)

        _attempt_adjacent_swaps(
            replicas,
            params,
            torch.tensor([1.0, 0.5], dtype=torch.float64),
            parity=0,
            swap_attempts=attempts,
            swap_accepts=accepts,
        )

        self.assertEqual(attempts.tolist(), [4])
        self.assertEqual(accepts.tolist(), [4])
        self.assertTrue(torch.all(replicas[0, :, 0, 1] == 1))
        self.assertTrue(torch.all(replicas[1, :, 0, 0] == 1))

    def test_swap_ratio_moves_favorable_low_energy_state_to_cold_slot(self):
        params = zero_model(L=1)
        params["bias"][0, 1] = 100.0
        replicas = torch.zeros((2, 1, 1, 2), dtype=torch.float64)
        replicas[0, 0, 0, 0] = 1
        replicas[1, 0, 0, 1] = 1
        attempts = torch.zeros(1, dtype=torch.int64)
        accepts = torch.zeros(1, dtype=torch.int64)

        _attempt_adjacent_swaps(
            replicas,
            params,
            torch.tensor([1.0, 0.5], dtype=torch.float64),
            parity=0,
            swap_attempts=attempts,
            swap_accepts=accepts,
        )

        self.assertEqual(accepts.tolist(), [1])
        self.assertEqual(replicas[0].argmax(dim=-1).item(), 1)

    def test_returns_exact_number_of_cold_slot_samples_and_diagnostics(self):
        torch.manual_seed(3)
        params = zero_model()
        progress = []
        result = replica_exchange_sampling(
            params,
            betas=[1.0, 0.6, 0.3],
            nsamples=5,
            nchains=2,
            burn_in_sweeps=2,
            sweeps_between_samples=2,
            progress=lambda completed, total: progress.append((completed, total)),
        )

        self.assertEqual(result.samples.shape, (5, 3, 2))
        self.assertEqual(result.energies.shape, (5,))
        self.assertEqual(result.final_replicas.shape, (3, 2, 3, 2))
        self.assertTrue(torch.all(result.samples.sum(dim=-1) == 1))
        torch.testing.assert_close(result.energies, compute_energy(result.samples, params))
        self.assertEqual(progress[-1], (8, 8))
        self.assertEqual(len(progress), 8)
        self.assertEqual(result.swap_attempts.tolist(), [6, 6])
        self.assertEqual(result.swap_accepts.tolist(), [6, 6])
        torch.testing.assert_close(result.swap_acceptance_rates, torch.ones(2, dtype=torch.float64))

    def test_two_replica_pair_is_attempted_every_sweep(self):
        params = zero_model(L=1)
        result = replica_exchange_sampling(
            params,
            betas=[1.0, 0.5],
            nsamples=2,
            nchains=2,
            burn_in_sweeps=1,
            sweeps_between_samples=1,
        )

        self.assertEqual(result.swap_attempts.tolist(), [2])
        self.assertEqual(result.swap_accepts.tolist(), [2])

    def test_adaptation_shrinks_low_acceptance_log_beta_gap(self):
        betas = torch.tensor([1.0, 0.5, 0.25], dtype=torch.float64)
        adapted, rates = _adapt_beta_ladder(
            betas,
            swap_attempts=torch.tensor([100, 100]),
            swap_accepts=torch.tensor([90, 0]),
            learning_rate=0.5,
        )

        old_gaps = torch.log(betas[:-1]) - torch.log(betas[1:])
        new_gaps = torch.log(adapted[:-1]) - torch.log(adapted[1:])
        self.assertGreater(float(rates[0]), float(rates[1]))
        self.assertGreater(float(new_gaps[0]), float(old_gaps[0]))
        self.assertLess(float(new_gaps[1]), float(old_gaps[1]))
        torch.testing.assert_close(adapted[[0, -1]], betas[[0, -1]])

    def test_adaptation_is_discarded_and_production_uses_frozen_ladder(self):
        params = zero_model()
        result = replica_exchange_sampling(
            params,
            betas=[1.0, 0.6, 0.3],
            nsamples=4,
            nchains=2,
            adaptation_sweeps=4,
            adaptation_window=2,
            burn_in_sweeps=3,
            sweeps_between_samples=1,
        )

        self.assertEqual(len(result.adaptation_history), 2)
        torch.testing.assert_close(result.betas, result.initial_betas)
        self.assertEqual(result.swap_attempts.tolist(), [2, 2])
        self.assertEqual(result.swap_accepts.tolist(), [2, 2])
        self.assertEqual(result.samples.shape[0], 4)

    def test_rejects_non_descending_beta_ladder(self):
        with self.assertRaisesRegex(ValueError, "strictly decreasing"):
            replica_exchange_sampling(zero_model(), [0.5, 1.0], nsamples=2)

    def test_cli_writes_flood_fill_compatible_fasta_and_swap_log(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            params_path = root / "params.dat"
            output_path = root / "output"
            save_params(str(params_path), zero_model(L=2), tokens="AB")

            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "adabmDCA.cli",
                    "replica-exchange",
                    "--path_params",
                    str(params_path),
                    "--output",
                    str(output_path),
                    "--ngen",
                    "3",
                    "--alphabet",
                    "AB",
                    "--nchains",
                    "2",
                    "--nreplicas",
                    "2",
                    "--burn_in_sweeps",
                    "0",
                    "--adaptation_sweeps",
                    "4",
                    "--adaptation_window",
                    "2",
                    "--sweeps_between_samples",
                    "1",
                    "--device",
                    "cpu",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            fasta_path = output_path / "samples.fasta"
            log_path = output_path / "replica_exchange.log"
            adaptation_log_path = output_path / "ladder_adaptation.log"
            self.assertTrue(fasta_path.is_file())
            self.assertTrue(log_path.is_file())
            self.assertTrue(adaptation_log_path.is_file())
            samples, = load_chains(str(fasta_path), tokens="AB", device=torch.device("cpu"))
            self.assertEqual(samples.shape, (3, 2, 2))
            self.assertIn("acceptance_rate", log_path.read_text())
            self.assertIn("smoothed_acceptance_rate", adaptation_log_path.read_text())


if __name__ == "__main__":
    unittest.main()
