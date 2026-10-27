# `adabmDCA.replica_exchange`

Replica-exchange (parallel-tempering) sampling for DCA models.

## `geometric_beta_ladder`

```python
geometric_beta_ladder(
    nreplicas: int,
    beta_min: float,
    beta_max: float = 1.0,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor
```

Constructs a geometrically spaced, strictly decreasing inverse-temperature
ladder. The first entry is the coldest, target temperature.

## `replica_exchange_sampling`

```python
replica_exchange_sampling(
    params,
    betas,
    nsamples,
    *,
    nchains=32,
    burn_in_sweeps=100,
    sweeps_between_samples=10,
    adaptation_sweeps=0,
    adaptation_window=100,
    adaptation_rate=0.5,
    adaptation_decay=0.5,
    sampling_method="gibbs",
    initial_replicas=None,
    progress=None,
) -> ReplicaExchangeResult
```

Evolves `nchains` independent replica ladders. Each sweep applies the selected
local MCMC kernel at every beta and then attempts a non-overlapping set of
adjacent exchanges. Even and odd exchange pairs alternate. Optional adaptation
redistributes log-beta gaps toward equal adjacent acceptance while holding both
endpoints fixed. Adaptive states are discarded, the ladder is frozen, and a
separate burn-in runs before maximum-beta configurations are collected.

`ReplicaExchangeResult` contains:

- `samples`: exactly `nsamples` one-hot target-temperature configurations;
- `energies`: corresponding unscaled DCA energies;
- `betas`: the validated beta ladder;
- `initial_betas` and detailed `adaptation_history`;
- `swap_attempts`, `swap_accepts`, and `swap_acceptance_rates`;
- `final_replicas`: the final state of every ladder and temperature slot.

The optional progress callback receives `(completed_sweeps, total_sweeps)`.
Replica identity is not fixed to a temperature: configurations move between
slots when exchanges are accepted.
