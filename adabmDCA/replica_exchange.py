"""Replica-exchange (parallel-tempering) sampling for DCA models."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
from torch.nn.functional import one_hot

from adabmDCA.sampling import get_sampler
from adabmDCA.statmech import compute_energy


@dataclass
class LadderAdaptationStep:
    """One completed temperature-ladder adaptation window."""

    sweep: int
    betas_before: torch.Tensor
    betas_after: torch.Tensor
    swap_attempts: torch.Tensor
    swap_accepts: torch.Tensor
    acceptance_rates: torch.Tensor


@dataclass
class ReplicaExchangeResult:
    """Samples and diagnostics from a replica-exchange run.

    Samples are configurations observed in the first (largest-beta,
    lowest-temperature) slot. Replica identities may change at swap steps.
    """

    samples: torch.Tensor
    energies: torch.Tensor
    betas: torch.Tensor
    swap_attempts: torch.Tensor
    swap_accepts: torch.Tensor
    final_replicas: torch.Tensor
    initial_betas: torch.Tensor
    adaptation_history: list[LadderAdaptationStep]

    @property
    def swap_acceptance_rates(self) -> torch.Tensor:
        """Acceptance fraction for every adjacent pair in the beta ladder."""
        rates = self.swap_accepts.to(self.betas.dtype) / self.swap_attempts.clamp_min(1)
        return torch.where(
            self.swap_attempts > 0,
            rates,
            torch.full_like(rates, torch.nan),
        )


def geometric_beta_ladder(
    nreplicas: int,
    beta_min: float,
    beta_max: float = 1.0,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Construct a geometrically spaced, descending inverse-temperature ladder."""
    if nreplicas < 2:
        raise ValueError("nreplicas must be at least 2.")
    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("beta_min and beta_max must be positive.")
    if beta_min >= beta_max:
        raise ValueError("beta_min must be smaller than beta_max.")
    return torch.logspace(
        torch.log10(torch.tensor(beta_max)).item(),
        torch.log10(torch.tensor(beta_min)).item(),
        steps=nreplicas,
        device=device,
        dtype=dtype,
    )


def _validate_inputs(
    params: Dict[str, torch.Tensor],
    betas: Sequence[float] | torch.Tensor,
    nsamples: int,
    nchains: int,
    burn_in_sweeps: int,
    sweeps_between_samples: int,
    adaptation_sweeps: int,
    adaptation_window: int,
    adaptation_rate: float,
    adaptation_decay: float,
) -> torch.Tensor:
    if "bias" not in params or "coupling_matrix" not in params:
        raise KeyError("params must contain 'bias' and 'coupling_matrix'.")
    bias = params["bias"]
    coupling = params["coupling_matrix"]
    if bias.dim() != 2:
        raise ValueError("params['bias'] must have shape (L, q).")
    L, q = bias.shape
    if coupling.shape != (L, q, L, q):
        raise ValueError("params['coupling_matrix'] must have shape (L, q, L, q).")
    if coupling.device != bias.device or coupling.dtype != bias.dtype:
        raise ValueError("Biases and couplings must share a device and dtype.")
    if nsamples < 1:
        raise ValueError("nsamples must be positive.")
    if nchains < 1:
        raise ValueError("nchains must be positive.")
    if burn_in_sweeps < 0:
        raise ValueError("burn_in_sweeps must be non-negative.")
    if sweeps_between_samples < 1:
        raise ValueError("sweeps_between_samples must be positive.")
    if adaptation_sweeps < 0:
        raise ValueError("adaptation_sweeps must be non-negative.")
    if adaptation_window < 1:
        raise ValueError("adaptation_window must be positive.")
    if adaptation_rate <= 0:
        raise ValueError("adaptation_rate must be positive.")
    if adaptation_decay < 0:
        raise ValueError("adaptation_decay must be non-negative.")

    beta_tensor = torch.as_tensor(betas, device=bias.device, dtype=bias.dtype)
    if beta_tensor.dim() != 1 or len(beta_tensor) < 2:
        raise ValueError("betas must be a one-dimensional ladder with at least two entries.")
    if not torch.all(torch.isfinite(beta_tensor)) or not torch.all(beta_tensor > 0):
        raise ValueError("betas must be finite and positive.")
    if not torch.all(beta_tensor[:-1] > beta_tensor[1:]):
        raise ValueError("betas must be strictly decreasing; the lowest-temperature slot comes first.")
    if adaptation_sweeps > 0 and len(beta_tensor) > 2 and adaptation_window < 2:
        raise ValueError("adaptation_window must be at least 2 when adapting more than two replicas.")
    if adaptation_sweeps == 1 and len(beta_tensor) > 2:
        raise ValueError("adaptation_sweeps must be at least 2 when adapting more than two replicas.")
    return beta_tensor.clone()


def _adapt_beta_ladder(
    betas: torch.Tensor,
    swap_attempts: torch.Tensor,
    swap_accepts: torch.Tensor,
    learning_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Redistribute fixed-endpoint log-beta gaps toward uniform acceptance."""
    if bool(torch.any(swap_attempts == 0)):
        raise ValueError("Every adjacent pair must be attempted before adapting the ladder.")
    rates = (swap_accepts.to(betas.dtype) + 1.0) / (swap_attempts.to(betas.dtype) + 2.0)
    log_betas = torch.log(betas)
    gaps = log_betas[:-1] - log_betas[1:]
    adjusted_gaps = gaps * torch.exp(learning_rate * (rates - rates.mean()))
    adjusted_gaps *= gaps.sum() / adjusted_gaps.sum()
    adapted_log_betas = torch.cat(
        [log_betas[:1], log_betas[:1] - torch.cumsum(adjusted_gaps, dim=0)]
    )
    adapted = torch.exp(adapted_log_betas)
    adapted[0] = betas[0]
    adapted[-1] = betas[-1]
    return adapted, rates


def _attempt_adjacent_swaps(
    replicas: torch.Tensor,
    params: Dict[str, torch.Tensor],
    betas: torch.Tensor,
    parity: int,
    swap_attempts: torch.Tensor,
    swap_accepts: torch.Tensor,
) -> None:
    """Attempt one non-overlapping set of adjacent swaps in place."""
    nreplicas, nchains = replicas.shape[:2]
    energies = compute_energy(replicas.reshape(-1, *replicas.shape[2:]), params).reshape(nreplicas, nchains)
    for lower_index in range(parity, nreplicas - 1, 2):
        upper_index = lower_index + 1
        log_ratio = (
            (betas[lower_index] - betas[upper_index])
            * (energies[lower_index] - energies[upper_index])
        )
        log_uniform = torch.log(torch.rand(nchains, device=replicas.device, dtype=replicas.dtype))
        swap_mask = log_uniform < log_ratio
        swap_attempts[lower_index] += nchains
        swap_accepts[lower_index] += swap_mask.sum().to(swap_accepts.device)
        if bool(swap_mask.any()):
            previous_lower = replicas[lower_index, swap_mask].clone()
            replicas[lower_index, swap_mask] = replicas[upper_index, swap_mask]
            replicas[upper_index, swap_mask] = previous_lower


@torch.no_grad()
def replica_exchange_sampling(
    params: Dict[str, torch.Tensor],
    betas: Sequence[float] | torch.Tensor,
    nsamples: int,
    *,
    nchains: int = 32,
    burn_in_sweeps: int = 100,
    sweeps_between_samples: int = 10,
    adaptation_sweeps: int = 0,
    adaptation_window: int = 100,
    adaptation_rate: float = 0.5,
    adaptation_decay: float = 0.5,
    sampling_method: str = "gibbs",
    initial_replicas: Optional[torch.Tensor] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ReplicaExchangeResult:
    """Sample the lowest-temperature slot of a replica-exchange ensemble.

    Each of ``nchains`` independent ladders contains one configuration per beta.
    Optional adaptation redistributes log-beta gaps while keeping both endpoints
    fixed. Adaptive samples are discarded, the resulting ladder is frozen, and a
    separate fixed-ladder burn-in precedes collection from the coldest slot.

    Args:
        params: DCA biases and couplings.
        betas: Strictly decreasing inverse temperatures. The first entry is the
            target, lowest-temperature ensemble returned in ``samples``.
        nsamples: Number of target-temperature samples to return.
        nchains: Number of independent replica ladders evolved in parallel.
        burn_in_sweeps: Replica-exchange sweeps discarded before collection.
        sweeps_between_samples: Sweeps between observations of the target slot.
        adaptation_sweeps: Number of discarded sweeps during which the beta
            ladder is adapted. Zero disables adaptation.
        adaptation_window: Swap-statistics window between ladder updates.
        adaptation_rate: Initial adaptation learning rate.
        adaptation_decay: Power-law decay exponent applied across windows.
        sampling_method: Local transition kernel, ``"gibbs"`` or ``"metropolis"``.
        initial_replicas: Optional tensor of shape ``(nreplicas, nchains, L, q)``.
        progress: Optional callback receiving ``(completed_sweeps, total_sweeps)``.
    """
    beta_tensor = _validate_inputs(
        params,
        betas,
        nsamples,
        nchains,
        burn_in_sweeps,
        sweeps_between_samples,
        adaptation_sweeps,
        adaptation_window,
        adaptation_rate,
        adaptation_decay,
    )
    sampler = get_sampler(sampling_method)
    nreplicas = len(beta_tensor)
    L, q = params["bias"].shape
    expected_shape = (nreplicas, nchains, L, q)
    if initial_replicas is None:
        categorical = torch.randint(0, q, (nreplicas, nchains, L), device=params["bias"].device)
        replicas = one_hot(categorical, num_classes=q).to(params["bias"].dtype)
    else:
        if initial_replicas.shape != expected_shape:
            raise ValueError(f"initial_replicas must have shape {expected_shape}.")
        if initial_replicas.device != params["bias"].device:
            raise ValueError("initial_replicas and params must be on the same device.")
        replicas = initial_replicas.to(params["bias"].dtype).clone()
        if not torch.all((replicas == 0) | (replicas == 1)) or not torch.all(replicas.sum(-1) == 1):
            raise ValueError("initial_replicas must be one-hot encoded.")

    batches_needed = (nsamples + nchains - 1) // nchains
    collection_sweeps = batches_needed * sweeps_between_samples
    total_sweeps = adaptation_sweeps + burn_in_sweeps + collection_sweeps
    initial_betas = beta_tensor.clone()
    adaptation_history: list[LadderAdaptationStep] = []
    collected: list[torch.Tensor] = []
    parity = 0
    completed_sweeps = 0

    def perform_sweep(attempts: torch.Tensor, accepts: torch.Tensor) -> None:
        nonlocal parity, completed_sweeps
        for replica_index, beta in enumerate(beta_tensor):
            replicas[replica_index] = sampler(
                chains=replicas[replica_index],
                params=params,
                nsweeps=1,
                beta=float(beta),
            )
        _attempt_adjacent_swaps(
            replicas,
            params,
            beta_tensor,
            parity,
            attempts,
            accepts,
        )
        parity = 0 if nreplicas == 2 else 1 - parity
        completed_sweeps += 1
        if progress is not None:
            progress(completed_sweeps, total_sweeps)

    adaptation_attempts = torch.zeros(nreplicas - 1, dtype=torch.int64, device=replicas.device)
    adaptation_accepts = torch.zeros_like(adaptation_attempts)
    window_start = 0
    window_index = 0
    for adaptation_sweep in range(adaptation_sweeps):
        perform_sweep(adaptation_attempts, adaptation_accepts)
        window_complete = adaptation_sweep + 1 - window_start >= adaptation_window
        adaptation_complete = adaptation_sweep + 1 == adaptation_sweeps
        if window_complete or adaptation_complete:
            if bool(torch.all(adaptation_attempts > 0)):
                betas_before = beta_tensor.clone()
                learning_rate = adaptation_rate / ((window_index + 1) ** adaptation_decay)
                beta_tensor, rates = _adapt_beta_ladder(
                    beta_tensor,
                    adaptation_attempts,
                    adaptation_accepts,
                    learning_rate,
                )
                adaptation_history.append(
                    LadderAdaptationStep(
                        sweep=adaptation_sweep + 1,
                        betas_before=betas_before,
                        betas_after=beta_tensor.clone(),
                        swap_attempts=adaptation_attempts.clone(),
                        swap_accepts=adaptation_accepts.clone(),
                        acceptance_rates=rates,
                    )
                )
                window_index += 1
            adaptation_attempts.zero_()
            adaptation_accepts.zero_()
            window_start = adaptation_sweep + 1

    discarded_attempts = torch.zeros(nreplicas - 1, dtype=torch.int64, device=replicas.device)
    discarded_accepts = torch.zeros_like(discarded_attempts)
    for _ in range(burn_in_sweeps):
        perform_sweep(discarded_attempts, discarded_accepts)

    swap_attempts = torch.zeros(nreplicas - 1, dtype=torch.int64, device=replicas.device)
    swap_accepts = torch.zeros_like(swap_attempts)
    for collection_sweep in range(collection_sweeps):
        perform_sweep(swap_attempts, swap_accepts)
        if (collection_sweep + 1) % sweeps_between_samples == 0:
            collected.append(replicas[0].clone())

    samples = torch.cat(collected, dim=0)[:nsamples]
    energies = compute_energy(samples, params)
    return ReplicaExchangeResult(
        samples=samples,
        energies=energies,
        betas=beta_tensor.clone(),
        swap_attempts=swap_attempts,
        swap_accepts=swap_accepts,
        final_replicas=replicas,
        initial_betas=initial_betas,
        adaptation_history=adaptation_history,
    )
