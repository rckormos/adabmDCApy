"""Command-line replica-exchange sampling workflow."""

import argparse
import csv

from adabmDCA.parser import add_args_replica_exchange
from adabmDCA.scripts._utils import ensure_output_dir, prefixed_path, require_file


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample a DCA model using replica-exchange MCMC."
    )
    return add_args_replica_exchange(parser)


def main() -> None:
    args = create_parser().parse_args()

    import torch
    from tqdm import tqdm

    from adabmDCA.fasta import get_tokens, write_fasta
    from adabmDCA.io import load_params
    from adabmDCA.replica_exchange import geometric_beta_ladder, replica_exchange_sampling
    from adabmDCA.utils import get_device, get_dtype

    require_file(args.path_params, "Parameters file")
    output_dir = ensure_output_dir(args.output)
    device = get_device(args.device)
    dtype = get_dtype(args.dtype)
    tokens = get_tokens(args.alphabet)
    params = load_params(args.path_params, tokens=tokens, device=device, dtype=dtype)

    if args.betas is None:
        betas = geometric_beta_ladder(
            nreplicas=args.nreplicas,
            beta_min=args.beta_min,
            beta_max=args.beta_max,
            device=device,
            dtype=dtype,
        )
    else:
        betas = torch.tensor(args.betas, device=device, dtype=dtype)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    batches_needed = (args.ngen + args.nchains - 1) // args.nchains
    total_sweeps = (
        args.adaptation_sweeps
        + args.burn_in_sweeps
        + batches_needed * args.sweeps_between_samples
    )
    print("\n" + "=" * 80)
    print("  REPLICA-EXCHANGE SAMPLING")
    print("=" * 80)
    print(f"  Parameters:              {args.path_params}")
    print(f"  Output directory:        {output_dir}")
    print(f"  Lowest-temperature beta: {float(betas[0]):.6g}")
    print(f"  Beta ladder:             {[round(float(beta), 6) for beta in betas]}")
    print(f"  Independent ladders:     {args.nchains}")
    print(f"  Requested samples:       {args.ngen}")
    print(f"  Burn-in sweeps:          {args.burn_in_sweeps}")
    print(f"  Adaptation sweeps:       {args.adaptation_sweeps}")
    if args.adaptation_sweeps:
        print(f"  Adaptation window:       {args.adaptation_window}")
        print(f"  Adaptation rate/decay:   {args.adaptation_rate}/{args.adaptation_decay}")
    print(f"  Sweeps between samples:  {args.sweeps_between_samples}")
    print(f"  Device:                  {device}\n")

    with tqdm(total=total_sweeps, desc="Replica exchange", unit="sweep") as progress_bar:
        def update_progress(completed: int, total: int) -> None:
            del total
            progress_bar.update(completed - progress_bar.n)

        result = replica_exchange_sampling(
            params=params,
            betas=betas,
            nsamples=args.ngen,
            nchains=args.nchains,
            burn_in_sweeps=args.burn_in_sweeps,
            sweeps_between_samples=args.sweeps_between_samples,
            adaptation_sweeps=args.adaptation_sweeps,
            adaptation_window=args.adaptation_window,
            adaptation_rate=args.adaptation_rate,
            adaptation_decay=args.adaptation_decay,
            sampling_method=args.sampler,
            progress=update_progress,
        )

    fasta_path = prefixed_path(output_dir, args.label, "samples.fasta")
    headers = [
        f"sequence {index + 1} | DCAenergy: {float(energy):.6f} | beta: {float(result.betas[0]):.6g}"
        for index, energy in enumerate(result.energies)
    ]
    write_fasta(
        fname=str(fasta_path),
        headers=headers,
        sequences=result.samples,
        tokens=tokens,
    )

    log_path = prefixed_path(output_dir, args.label, "replica_exchange.log")
    with log_path.open("w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["beta_cold", "beta_hot", "attempts", "accepts", "acceptance_rate"])
        for pair_index, rate in enumerate(result.swap_acceptance_rates):
            writer.writerow(
                [
                    float(result.betas[pair_index]),
                    float(result.betas[pair_index + 1]),
                    int(result.swap_attempts[pair_index]),
                    int(result.swap_accepts[pair_index]),
                    float(rate),
                ]
            )

    adaptation_log_path = None
    if result.adaptation_history:
        adaptation_log_path = prefixed_path(output_dir, args.label, "ladder_adaptation.log")
        with adaptation_log_path.open("w", newline="") as log_file:
            writer = csv.writer(log_file)
            writer.writerow(
                [
                    "adaptation_sweep",
                    "pair_index",
                    "beta_cold_before",
                    "beta_hot_before",
                    "beta_cold_after",
                    "beta_hot_after",
                    "attempts",
                    "accepts",
                    "smoothed_acceptance_rate",
                ]
            )
            for step in result.adaptation_history:
                for pair_index, rate in enumerate(step.acceptance_rates):
                    writer.writerow(
                        [
                            step.sweep,
                            pair_index,
                            float(step.betas_before[pair_index]),
                            float(step.betas_before[pair_index + 1]),
                            float(step.betas_after[pair_index]),
                            float(step.betas_after[pair_index + 1]),
                            int(step.swap_attempts[pair_index]),
                            int(step.swap_accepts[pair_index]),
                            float(rate),
                        ]
                    )

    if result.adaptation_history:
        print(f"\nAdapted beta ladder: {[round(float(beta), 6) for beta in result.betas]}")
    print("\nProduction swap acceptance rates:")
    for pair_index, rate in enumerate(result.swap_acceptance_rates):
        print(
            f"  beta {float(result.betas[pair_index]):.4g} ↔ "
            f"{float(result.betas[pair_index + 1]):.4g}: {float(rate):.3f}"
        )
    print(f"\nSamples: {fasta_path}")
    print(f"Swap log: {log_path}\n")
    if adaptation_log_path is not None:
        print(f"Adaptation log: {adaptation_log_path}\n")


if __name__ == "__main__":
    main()
