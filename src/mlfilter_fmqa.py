"""L-layer optical-filter FMQA closed loop (One-Hot, 4 materials per layer).

Generalizes the audited L=6 loop of Intern_Fujitsu/intern1016/multilayer_filter/fmqa_penalty
(trajectory/fit/sample) and fixed_fm (measure/select) to any layer count L:
  - candidates: itertools.product(range(4), repeat=L); index = sum_g c_g * 4**(L-1-g)
  - each cycle: fresh FM (seed + 100000*(cycle-1)), S = max |Ising h, J| of the FM QUBO,
    lambda = S * alpha, Neal SA (100 reads x 1000 sweeps, seed*100000 + (cycle-1)*1000),
    select the minimum-FM new feasible candidate (ties by one-hot tuple, as before), evaluate it
  - invalid / no-candidate cycles consume a proposal cycle but no evaluation
Additional proposal rule 'RandomFM' (alpha -> infinity limit): 100 uniform feasible label
vectors per cycle instead of SA, same FM selection. The truth values come from a validated
exhaustive table; the loop only reads values of candidates it has selected.

Requires on sys.path: Intern_Fujitsu/src (fm, fm_to_qubo, fmqa_solver, qarp_backend),
Intern_Fujitsu root (intern_0924.src.penalty), and qarpx/neal/dimod.
"""
from __future__ import annotations

import hashlib
import random
import time

import numpy as np

TRAINING = {'rank': 2, 'optimizer': 'AdamW', 'weight_decay': 0.01, 'learning_rate': 0.1, 'epochs': 120}
SA = {'num_reads': 100, 'num_sweeps': 1000, 'beta_schedule_type': 'geometric'}
ADAPTIVE = {'initial_alpha': 1.0, 'min_alpha': 0.01, 'max_alpha': 100.0}
RANDOM_STREAM_OFFSET = 73101


def labels_for(layers):
    import itertools
    return np.array(list(itertools.product(range(4), repeat=layers)), dtype=np.int8)


def onehot(labels):
    a = np.asarray(labels)
    return np.eye(4, dtype=np.int8)[a].reshape(len(a), 4 * a.shape[1])


def initial_ids(seed, layers, count=40):
    return random.Random(seed).sample(range(4 ** layers), count)


def matrix(qdict, n):
    # Same as intern_1009/perovskites/fixed_fm_pilot/py/run_pilot.py:matrix (upper triangle).
    q = np.zeros((n, n))
    for (i, j), v in qdict.items():
        q[min(i, j), max(i, j)] += v
    return q


def model_hash(model):
    # Same as run_pilot.py:model_hash.
    h = hashlib.sha256()
    for k, v in model.state_dict().items():
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fit(x, y, seed):
    import torch
    from fm import train_factorization_machine
    from fm_to_qubo import fm_to_qubo
    from qarp_backend import qubo_to_ising_coefficients
    n = x.shape[1]
    torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    m = train_factorization_machine(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), n,
                                    k=TRAINING['rank'], epochs=TRAINING['epochs'], learning_rate=TRAINING['learning_rate'],
                                    optimizer_name=TRAINING['optimizer'], weight_decay=TRAINING['weight_decay'])
    m.eval(); qd, b = fm_to_qubo(m); q = matrix(qd, n)
    h, j = qubo_to_ising_coefficients(q)
    s = max([abs(v) for v in h.values()] + [abs(v) for v in j.values()] + [0.])
    return m, qd, q, b, s


def predict_all(x_all, q, b, chunk=1 << 15):
    out = np.empty(len(x_all))
    for first in range(0, len(x_all), chunk):
        xx = x_all[first:first + chunk].astype(np.float64)
        out[first:first + chunk] = np.einsum('bi,ij,bj->b', xx, q, xx) + b
    return out


def sample_sa(qd, b, s, alpha, layers, seed):
    import neal
    from fmqa_solver import build_penalty_bqm
    n = 4 * layers; lam = s * alpha
    result = neal.SimulatedAnnealingSampler().sample(build_penalty_bqm(qd, [4] * layers, lam, b), seed=seed, **SA)
    arr = result.record.sample[:, [list(result.variables).index(i) for i in range(n)]]
    raw = np.repeat(arr, result.record.num_occurrences, axis=0).astype(np.int8)
    assert raw.shape == (SA['num_reads'], n)
    return raw, {'alpha': alpha, 'lambda': lam, 'S': s, 'beta_range': list(result.info['beta_range']), 'sampler_seed': seed}


def sample_random_feasible(layers, seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, (SA['num_reads'], layers))
    return onehot(labels), {'alpha': None, 'lambda': None, 'S': None, 'sampler_seed': seed, 'sampler': 'uniform feasible labels'}


def bits_to_ids(raw, layers):
    """Candidate index for each one-hot-valid row, None for invalid rows."""
    g = raw.reshape(len(raw), layers, 4)
    valid = (g.sum(2) == 1).all(1)
    idx = (g.argmax(2).astype(np.int64) * (4 ** np.arange(layers - 1, -1, -1))).sum(1)
    return [int(i) if v else None for i, v in zip(idx, valid)]


def select(raw, pred, x_all, seen, layers):
    """Same rule as intern_1009/.../fmqa_adamw_bbo/py/run_bbo.py:select, without a full pattern dict."""
    ids = bits_to_ids(raw, layers)
    unseen = {i for i in ids if i is not None and i not in seen}
    chosen = min(unseen, key=lambda i: (pred[i], tuple(x_all[i]))) if unseen else None
    return chosen, {'raw_feasible_rate': sum(i is not None for i in ids) / len(ids),
                    'invalid_count': sum(i is None for i in ids),
                    'evaluated_duplicate_count': sum(i in seen for i in ids if i is not None),
                    'unique_unseen_count': len(unseen),
                    'within_unseen_duplicate_count': sum(i is not None and i not in seen for i in ids) - len(unseen)}


def trajectory(seed, condition, layers, truth, budget, cycle_cap, x_all=None):
    """Run one FMQA (or RandomFM) closed loop. condition: {'id', 'kind': 'sa'|'randomfm', 'alpha', 'adaptive'}."""
    from intern_0924.src.penalty import AdaptivePenaltyTracker
    x_all = onehot(labels_for(layers)) if x_all is None else x_all
    ids = initial_ids(seed, layers); ys = [float(truth[i]) for i in ids]
    tracker = AdaptivePenaltyTracker(**ADAPTIVE)
    events = []; start = time.perf_counter()
    for cycle in range(1, cycle_cap + 1):
        if len(ids) >= budget:
            break
        modelseed = seed + 100000 * (cycle - 1)
        t = time.perf_counter(); m, qd, q, b, s = fit(x_all[ids], ys, modelseed); training = time.perf_counter() - t
        pred = predict_all(x_all, q, b)
        sampler_seed = seed * 100000 + (cycle - 1) * 1000
        t = time.perf_counter()
        if condition['kind'] == 'randomfm':
            raw, meta = sample_random_feasible(layers, sampler_seed)
        else:
            alpha = tracker.get_current_alpha() if condition.get('adaptive') else condition['alpha']
            raw, meta = sample_sa(qd, b, s, alpha, layers, sampler_seed)
        sampling = time.perf_counter() - t
        seen = set(ids)
        chosen, filters = select(raw, pred, x_all, seen, layers)
        unseen_mask = np.ones(len(pred), bool); unseen_mask[ids] = False
        unseen_min = float(pred[unseen_mask].min())
        event = {'cycle': cycle, 'model_seed': modelseed, 'model_sha256': model_hash(m), 'S_fm': s, **meta, **filters,
                 'selected_id': chosen, 'selected_value': None,
                 'selected_FM_gap': None if chosen is None else float(pred[chosen] - unseen_min),
                 'unseen_FM_min_hit': bool(chosen is not None and abs(pred[chosen] - unseen_min) < 1e-8),
                 'evaluations_before': len(ids), 'training_seconds': training, 'sampling_seconds': sampling}
        if condition.get('adaptive'):
            event['next_alpha'] = tracker.update(filters['raw_feasible_rate'])
        if chosen is not None:
            assert chosen not in seen and len(ids) < budget
            value = float(truth[chosen]); ids.append(chosen); ys.append(value); event['selected_value'] = value
        event.update(evaluations_after=len(ids), best_so_far=float(min(ys)))
        events.append(event)
    return {'seed': seed, 'condition': condition, 'layers': layers, 'budget': budget, 'cycle_cap': cycle_cap,
            'initial_ids': ids[:40], 'final_ids': ids, 'final_values': ys, 'actual_evaluations': len(ids),
            'final_best': float(min(ys)), 'events': events, 'total_seconds': time.perf_counter() - start,
            'stopping_reason': 'evaluation_cap' if len(ids) >= budget else 'proposal_cycle_cap'}


def random_run(seed, layers, truth, budget):
    """Uniform without replacement from unobserved candidates (same stream rule as mlfilter-pen30)."""
    ids = initial_ids(seed, layers); seen = set(ids)
    remaining = [i for i in range(4 ** layers) if i not in seen]
    choices = random.Random(seed + RANDOM_STREAM_OFFSET).sample(remaining, budget - len(ids))
    ids = ids + choices; ys = [float(truth[i]) for i in ids]
    return {'seed': seed, 'condition': {'id': 'Random', 'kind': 'random'}, 'layers': layers, 'budget': budget,
            'initial_ids': ids[:40], 'final_ids': ids, 'final_values': ys, 'actual_evaluations': len(ids),
            'final_best': float(min(ys)), 'events': [], 'stopping_reason': 'evaluation_cap'}
