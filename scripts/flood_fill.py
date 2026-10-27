"""Enumerate low-energy sequence components from seed sequences."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"file does not exist: {path}")
    return path


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "unlimited"}:
        return None
    return _positive_int(value)


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _percentile(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not math.isfinite(parsed) or not 0 <= parsed <= 100:
        raise argparse.ArgumentTypeError("must be between 0 and 100")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Enumerate connected components of sequences below a DCA-energy "
            "threshold, starting from seed sequences in a FASTA file."
        )
    )
    parser.add_argument(
        "-p", "--params", "--path-params", dest="params", required=True,
        type=_existing_file, metavar="FILE", help="DCA parameter file.",
    )
    parser.add_argument(
        "-s", "--seeds", "--seed-fasta", dest="seeds", required=True,
        type=_existing_file, metavar="FILE", help="FASTA file containing seed sequences.",
    )
    parser.add_argument(
        "-o", "--output", required=True, type=Path, metavar="FILE",
        help="Destination FASTA file (its parent directory is created if needed).",
    )
    threshold_group = parser.add_mutually_exclusive_group(required=True)
    threshold_group.add_argument(
        "-t", "--energy-threshold", type=float, metavar="ENERGY",
        help="Absolute DCA-energy boundary for accepted sequences.",
    )
    threshold_group.add_argument(
        "--energy-percentile", "--percentile", type=_percentile, metavar="PERCENT",
        help=(
            "Set the energy boundary to this percentile of the energies of all "
            "input seed samples (0 to 100)."
        ),
    )
    parser.add_argument(
        "--alphabet", default="protein",
        help="Built-in alphabet (protein, DNA, or RNA) or a custom token string (default: protein).",
    )
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32",
        help="Floating-point precision used for the model (default: float32).",
    )
    parser.add_argument(
        "--max-states", type=_optional_positive_int, default=500_000, metavar="N|none",
        help="Maximum accepted states; use 'none' for no limit (default: 500000).",
    )
    parser.add_argument(
        "--max-frontier", type=_optional_positive_int, default=1_000_000, metavar="N|none",
        help="Maximum queued states; use 'none' for no limit (default: 1000000).",
    )
    parser.add_argument(
        "--max-gaps", type=_nonnegative_int, default=None, metavar="N",
        help="Maximum number of '-' characters allowed in a queued sequence (default: unlimited).",
    )
    parser.add_argument(
        "--boundary-tolerance", type=_nonnegative_float, default=1e-6, metavar="EPSILON",
        help="Exact-energy recomputation tolerance near the boundary (default: 1e-6).",
    )
    parser.add_argument(
        "--progress-interval", type=_positive_int, default=100_000, metavar="N",
        help="Number of tested states between progress updates (default: 100000).",
    )
    parser.add_argument(
        "--exclusive", action="store_true",
        help="Exclude sequences whose energy equals the threshold.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress progress updates (the summary is still printed).",
    )
    return parser


def _show_progress(progress: Any) -> None:
    frontier_limit = "unlimited" if progress.max_frontier is None else f"{progress.max_frontier:,}"
    print(
        f"\r\033[Ktested={progress.tested:,}  "
        f"unique_states={progress.accepted:,}  "
        f"frontier={progress.frontier:,}/{frontier_limit}  "
        f"unresolved_seeds={progress.unresolved_seeds:,}/{progress.eligible_seeds:,}",
        end="",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)

    import torch

    from adabmDCA import flood_fill_low_energy
    from adabmDCA.fasta import get_tokens, write_fasta
    from adabmDCA.io import load_chains, load_params
    from adabmDCA.statmech import compute_energy

    output = args.output.expanduser()
    if output.resolve() in {args.params.resolve(), args.seeds.resolve()}:
        parser.error("--output must not overwrite an input file")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        parser.error(f"cannot create output directory {output.parent}: {error}")

    dtype = getattr(torch, args.dtype)
    device = torch.device("cpu")
    tokens = get_tokens(args.alphabet)
    if args.max_gaps is not None and tokens.count("-") != 1:
        parser.error("--max-gaps requires an alphabet containing exactly one '-' gap character")
    gap_token = tokens.index("-") if "-" in tokens else 0
    params = load_params(fname=str(args.params), tokens=tokens, device=device, dtype=dtype)
    (seeds,) = load_chains(fname=str(args.seeds), tokens=tokens, device=device, dtype=dtype)

    if args.energy_percentile is None:
        energy_threshold = args.energy_threshold
        threshold_description = "explicit"
    else:
        if seeds.shape[0] == 0:
            parser.error("cannot calculate an energy percentile from an empty seed FASTA")
        # Keep the matrix products bounded for large MCMC/REMC sample files.
        sample_energies = torch.cat(
            [compute_energy(batch, params) for batch in seeds.split(256)]
        )
        energy_threshold = float(
            torch.quantile(sample_energies, args.energy_percentile / 100.0)
        )
        threshold_description = f"input sample percentile {args.energy_percentile:g}"

    print(f"Energy threshold: {energy_threshold:.6f} ({threshold_description})")

    result = flood_fill_low_energy(
        seeds=seeds,
        params=params,
        energy_threshold=energy_threshold,
        inclusive=not args.exclusive,
        max_states=args.max_states,
        max_frontier=args.max_frontier,
        max_gaps=args.max_gaps,
        gap_token=gap_token,
        boundary_tolerance=args.boundary_tolerance,
        progress=None if args.quiet else _show_progress,
        progress_interval=args.progress_interval,
    )
    if not args.quiet:
        print()

    headers = [
        f"state_{index} | DCAenergy: {energy:.6f} | component: {component}"
        for index, (energy, component) in enumerate(
            zip(result.energies.tolist(), result.component_ids.tolist())
        )
    ]
    write_fasta(fname=str(output), headers=headers, sequences=result.sequences, tokens=tokens)

    print(f"Eligible seeds: {result.eligible_seed_mask.sum().item()}/{len(seeds)}")
    print(f"Unique states: {len(result.sequences)}")
    minimum_energy = float(result.energies.min()) if result.energies.numel() else None
    print(f"Minimum energy: {minimum_energy:.6f}" if minimum_energy is not None else "Minimum energy: N/A")
    print(f"Components: {result.component_ids.unique().numel()}")
    print(f"Tested states: {result.num_tested}")
    print(f"Complete: {result.complete}")
    print(f"Termination reason: {result.termination_reason}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
