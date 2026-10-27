"""Exhaustive traversal of thresholded low-energy sequence-space components."""

from collections import deque
from dataclasses import dataclass
import sys
from typing import Callable, Dict, Optional

import torch
from torch.nn.functional import one_hot

from adabmDCA.statmech import compute_energy


@dataclass
class FloodFillResult:
    """Result of a low-energy flood-fill traversal.

    ``sequences`` uses categorical encoding, with shape ``(n_states, L)``.
    ``component_ids`` maps every returned state to a connected component, while
    ``seed_component_ids`` maps the input seeds to those components. A seed gets
    ``-1`` when it fails an eligibility constraint or was not reached before
    truncation.
    """

    sequences: torch.Tensor
    energies: torch.Tensor
    component_ids: torch.Tensor
    seed_component_ids: torch.Tensor
    eligible_seed_mask: torch.Tensor
    complete: bool
    termination_reason: str
    num_tested: int

    def to_one_hot(self, num_classes: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Return the discovered sequences in the library's one-hot format."""
        return one_hot(self.sequences, num_classes=num_classes).to(dtype)


@dataclass(frozen=True)
class FloodFillProgress:
    """Snapshot passed to a flood-fill progress callback."""

    tested: int
    accepted: int
    frontier: int
    max_frontier: Optional[int]
    components: int
    unresolved_seeds: int
    eligible_seeds: int
    complete: Optional[bool] = None
    termination_reason: Optional[str] = None


def _as_categorical(seeds: torch.Tensor, q: int) -> torch.Tensor:
    if seeds.dim() == 3:
        if seeds.shape[-1] != q:
            raise ValueError(f"Expected one-hot seeds with {q} states, got {seeds.shape[-1]}.")
        if not torch.all((seeds == 0) | (seeds == 1)) or not torch.all(seeds.sum(dim=-1) == 1):
            raise ValueError("One-hot seeds must contain exactly one active state per site.")
        seeds = seeds.argmax(dim=-1)
    elif seeds.dim() != 2:
        raise ValueError("Seeds must have shape (n_seeds, L) or (n_seeds, L, q).")

    if seeds.is_floating_point() and not torch.all(seeds == seeds.round()):
        raise ValueError("Categorical seeds must contain integer state indices.")
    seeds = seeds.to(torch.int64)
    if seeds.numel() and (seeds.min() < 0 or seeds.max() >= q):
        raise ValueError(f"Categorical seed states must be in [0, {q - 1}].")
    return seeds


def _state_key(state: torch.Tensor, q: int) -> bytes:
    """Return an exact, compact key for a categorical CPU sequence.

    Common DCA alphabets fit in one byte per site. Larger custom alphabets fall
    back to the tensor's int64 byte representation. Unlike
    ``tuple(state.tolist())``, neither representation allocates one Python
    object slot per site.
    """
    if q <= 256:
        return bytes(state.tolist())
    return state.contiguous().numpy().tobytes()


def _write_key_state(buffer: bytearray, site: int, value: int, q: int) -> None:
    """Replace one categorical state in a reusable encoded-key buffer."""
    if q <= 256:
        buffer[site] = value
        return
    start = site * 8
    buffer[start : start + 8] = value.to_bytes(8, byteorder=sys.byteorder, signed=True)


def _validate_cpu_model(params: Dict[str, torch.Tensor]) -> tuple[int, int]:
    if "bias" not in params or "coupling_matrix" not in params:
        raise KeyError("params must contain 'bias' and 'coupling_matrix'.")
    bias = params["bias"]
    coupling = params["coupling_matrix"]
    if bias.device.type != "cpu" or coupling.device.type != "cpu":
        raise ValueError("The initial flood-fill implementation supports CPU tensors only.")
    if bias.dim() != 2:
        raise ValueError("params['bias'] must have shape (L, q).")
    L, q = bias.shape
    if coupling.shape != (L, q, L, q):
        raise ValueError("params['coupling_matrix'] must have shape (L, q, L, q).")
    if bias.dtype != coupling.dtype:
        raise ValueError("Biases and couplings must have the same dtype.")
    if not bias.is_floating_point():
        raise ValueError("Model parameters must use a floating-point dtype.")

    transpose = coupling.permute(2, 3, 0, 1)
    if not torch.allclose(coupling, transpose, rtol=1e-5, atol=1e-7):
        raise ValueError("Incremental energies require a symmetric coupling matrix.")
    positions = torch.arange(L)
    self_couplings = coupling[positions, :, positions, :]
    if not torch.allclose(self_couplings, torch.zeros_like(self_couplings), rtol=0.0, atol=1e-7):
        raise ValueError("Incremental energies require zero within-site coupling blocks.")
    return L, q


def _local_fields(state: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Compute h_i(a) + sum_j J_ij(a, state_j) for one categorical state."""
    L = state.shape[0]
    positions = torch.arange(L)
    return params["bias"] + params["coupling_matrix"][:, :, positions, state].sum(dim=-1)


def _exact_energy(state: torch.Tensor, params: Dict[str, torch.Tensor], q: int) -> torch.Tensor:
    encoded = one_hot(state.unsqueeze(0), num_classes=q).to(params["bias"].dtype)
    return compute_energy(encoded, params)[0]


@torch.no_grad()
def flood_fill_low_energy(
    seeds: torch.Tensor,
    params: Dict[str, torch.Tensor],
    energy_threshold: float,
    *,
    inclusive: bool = True,
    max_states: Optional[int] = 100_000,
    max_frontier: Optional[int] = 100_000,
    max_gaps: Optional[int] = None,
    gap_token: int = 0,
    boundary_tolerance: float = 1e-6,
    progress: Optional[Callable[[FloodFillProgress], None]] = None,
    progress_interval: int = 10_000,
) -> FloodFillResult:
    """Enumerate low-energy components adjoining one or more seed sequences.

    Two sequences are adjacent when they differ at exactly one site. Only states
    whose DCA energy passes ``energy_threshold`` and whose gap count passes the
    optional constraint are included. Eligible seeds in the same component are
    deduplicated and mapped to the same component ID.

    The traversal is exhaustive only when the returned ``complete`` flag is true.
    Safety limits bound accidental exploration of a very large component; pass
    ``None`` to disable either limit. Energy changes are evaluated incrementally,
    with exact recomputation for candidates close to the threshold.

    Args:
        seeds: Categorical ``(n_seeds, L)`` or one-hot ``(n_seeds, L, q)`` CPU tensor.
        params: CPU Potts parameters with symmetric, zero-self couplings.
        energy_threshold: Absolute DCA-energy boundary.
        inclusive: Include states exactly at the boundary when true.
        max_states: Maximum number of accepted states across all components.
        max_frontier: Maximum number of accepted states waiting to be expanded.
        max_gaps: Maximum number of occurrences of ``gap_token`` in any state.
            States exceeding the limit are not added to the traversal. ``None``
            disables the constraint.
        gap_token: Categorical state index representing a gap.
        boundary_tolerance: Recompute candidate energies exactly within this
            absolute distance of the threshold.
        progress: Optional callback receiving progress snapshots. The callback
            is invoked after each ``progress_interval`` newly tested states and
            once when traversal terminates.
        progress_interval: Number of newly tested states between callbacks.
    """
    L, q = _validate_cpu_model(params)
    if seeds.device.type != "cpu":
        raise ValueError("The initial flood-fill implementation supports CPU tensors only.")
    states = _as_categorical(seeds, q)
    if states.shape[1] != L:
        raise ValueError(f"Seed length {states.shape[1]} does not match model length {L}.")
    if max_states is not None and max_states < 1:
        raise ValueError("max_states must be positive or None.")
    if max_frontier is not None and max_frontier < 1:
        raise ValueError("max_frontier must be positive or None.")
    if max_gaps is not None and max_gaps < 0:
        raise ValueError("max_gaps must be non-negative or None.")
    if not 0 <= gap_token < q:
        raise ValueError(f"gap_token must be in [0, {q - 1}].")
    if boundary_tolerance < 0:
        raise ValueError("boundary_tolerance must be non-negative.")
    if progress_interval < 1:
        raise ValueError("progress_interval must be positive.")

    threshold = torch.as_tensor(energy_threshold, dtype=params["bias"].dtype)

    def passes(value: torch.Tensor) -> bool:
        if inclusive:
            return bool(value <= threshold)
        return bool(value < threshold)

    n_seeds = states.shape[0]
    seed_energies = torch.empty(n_seeds, dtype=params["bias"].dtype)
    for index, state in enumerate(states):
        seed_energies[index] = _exact_energy(state, params, q)
    eligible_seed_mask = seed_energies <= threshold if inclusive else seed_energies < threshold
    if max_gaps is not None:
        eligible_seed_mask &= (states == gap_token).sum(dim=1) <= max_gaps
    seed_component_ids = torch.full((n_seeds,), -1, dtype=torch.int64)
    eligible_seed_keys = {
        _state_key(state, q)
        for state, eligible in zip(states, eligible_seed_mask)
        if bool(eligible)
    }
    unresolved_seed_keys = set(eligible_seed_keys)

    output_states: list[torch.Tensor] = []
    output_energies: list[torch.Tensor] = []
    output_components: list[int] = []
    # A negative value marks a tested, rejected state; non-negative values are
    # indices into the accepted-state output arrays. Keeping both cases in one
    # table avoids a second hash-table allocation while retaining exact lookup.
    rejected = -1
    visited: dict[bytes, int] = {}
    complete = True
    termination_reason = "frontier_exhausted"
    next_component = 0
    next_progress_at = progress_interval
    frontier = deque()

    def report_progress(frontier_size: int, *, final: bool = False) -> None:
        nonlocal next_progress_at
        if progress is None:
            return
        if not final and len(visited) < next_progress_at:
            return
        progress(
            FloodFillProgress(
                tested=len(visited),
                accepted=len(output_states),
                frontier=frontier_size,
                max_frontier=max_frontier,
                components=next_component,
                unresolved_seeds=len(unresolved_seed_keys),
                eligible_seeds=len(eligible_seed_keys),
                complete=complete if final else None,
                termination_reason=termination_reason if final else None,
            )
        )
        while next_progress_at <= len(visited):
            next_progress_at += progress_interval

    for seed_index, seed in enumerate(states):
        seed_key = _state_key(seed, q)
        if not bool(eligible_seed_mask[seed_index]):
            visited.setdefault(seed_key, rejected)
            report_progress(0)
            continue
        accepted_index = visited.get(seed_key, rejected)
        if accepted_index >= 0:
            seed_component_ids[seed_index] = output_components[accepted_index]
            continue
        if max_states is not None and len(output_states) >= max_states:
            complete = False
            termination_reason = "max_states"
            break

        component_id = next_component
        next_component += 1
        seed_component_ids[seed_index] = component_id
        seed_energy = seed_energies[seed_index].clone()
        seed_state = seed.clone()
        seed_gap_count = int((seed_state == gap_token).sum())
        visited[seed_key] = len(output_states)
        unresolved_seed_keys.discard(seed_key)
        output_states.append(seed_state)
        output_energies.append(seed_energy)
        output_components.append(component_id)
        frontier = deque(
            [(seed_state, seed_key, seed_energy, _local_fields(seed_state, params), seed_gap_count)]
        )

        while frontier:
            state, state_key, energy, fields, gap_count = frontier.popleft()
            key_buffer = bytearray(state_key)
            for site in range(L):
                old_state = int(state[site])
                child_energies = energy + fields[site, old_state] - fields[site]
                for new_state in range(q):
                    if new_state == old_state:
                        continue
                    _write_key_state(key_buffer, site, new_state, q)
                    child_key = bytes(key_buffer)
                    if child_key in visited:
                        continue
                    visited[child_key] = rejected

                    child_gap_count = gap_count
                    if old_state == gap_token:
                        child_gap_count -= 1
                    if new_state == gap_token:
                        child_gap_count += 1
                    if max_gaps is not None and child_gap_count > max_gaps:
                        report_progress(len(frontier))
                        continue

                    child_energy = child_energies[new_state]
                    child = None
                    if abs(float(child_energy - threshold)) <= boundary_tolerance:
                        child = state.clone()
                        child[site] = new_state
                        child_energy = _exact_energy(child, params, q)
                    if not passes(child_energy):
                        report_progress(len(frontier))
                        continue
                    if max_states is not None and len(output_states) >= max_states:
                        complete = False
                        termination_reason = "max_states"
                        break
                    if max_frontier is not None and len(frontier) >= max_frontier:
                        complete = False
                        termination_reason = "max_frontier"
                        break

                    if child is None:
                        child = state.clone()
                        child[site] = new_state
                    visited[child_key] = len(output_states)
                    unresolved_seed_keys.discard(child_key)
                    output_states.append(child)
                    output_energies.append(child_energy.clone())
                    output_components.append(component_id)
                    child_fields = (
                        fields
                        + params["coupling_matrix"][:, :, site, new_state]
                        - params["coupling_matrix"][:, :, site, old_state]
                    )
                    frontier.append((child, child_key, child_energy, child_fields, child_gap_count))
                    report_progress(len(frontier))
                _write_key_state(key_buffer, site, old_state, q)
                if not complete:
                    break
            if not complete:
                break
        if not complete:
            break

    report_progress(len(frontier), final=True)

    if output_states:
        sequences_out = torch.stack(output_states)
        energies_out = torch.stack(output_energies)
    else:
        sequences_out = torch.empty((0, L), dtype=torch.int64)
        energies_out = torch.empty(0, dtype=params["bias"].dtype)

    return FloodFillResult(
        sequences=sequences_out,
        energies=energies_out,
        component_ids=torch.tensor(output_components, dtype=torch.int64),
        seed_component_ids=seed_component_ids,
        eligible_seed_mask=eligible_seed_mask,
        complete=complete,
        termination_reason=termination_reason,
        num_tested=len(visited),
    )
