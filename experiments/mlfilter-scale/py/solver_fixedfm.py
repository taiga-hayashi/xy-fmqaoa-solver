"""mlfilter-scale solver-level comparison on frozen FMs (exploratory).

  init    --layers L --truth-from <phase0 run>
  freeze  --run-dir D                      train and freeze the 5 FMs
  sa      --run-dir D --seed S             SA alpha sweep + RandomFM
  xy      --run-dir D --seed S --objective mean|cvar10     XY-FMQAOA p=1..3
  xstd    --run-dir D --seed S --alpha A   standard (X-mixer) QAOA with penalty, L=6 only
  analyze --run-dir D
"""
import argparse
import hashlib
import itertools
import json
import secrets
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
PROJECT = STUDY_DIR.parents[1]
CONFIG = json.loads((STUDY_DIR/'solver_fixedfm.json').read_text())
PATHS = json.loads((PROJECT/'config/paths.local.json').read_text())['profiles']['server']
INTERN = Path(PATHS['intern_fujitsu_root'])
sys.path[:0] = [str(PROJECT/'src'), str(INTERN/'src'), str(INTERN), PATHS['extra_pythonpath']]
import mlfilter_fmqa as M

CODE = [Path(__file__).resolve(), STUDY_DIR/'solver_fixedfm.json', PROJECT/'src/mlfilter_fmqa.py', PROJECT/'src/qarp_pair_xy.py',
        *[INTERN/p for p in ['src/fm.py','src/fm_to_qubo.py','src/fmqa_solver.py','src/bb_function.py','src/qarp_backend.py',
                             'intern_0924/src/solvers.py','intern_0924/src/penalty.py']]]
GRID = {'gammas': [0.0, 0.05, 0.2, 0.4, 0.8, 1.6, 3.2], 'betas': [0.0, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6]}  # mlfilter-fixedfm protocol


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    tmp = Path(str(path)+'.tmp'); tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n'); tmp.replace(path)


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def code_hashes():
    return {str(p):sha(p) for p in CODE}


def init(layers, truth_from):
    truth_from = truth_from.resolve(); enum = json.loads((truth_from/'raw/enumeration.json').read_text())
    assert enum['status']=='completed' and enum['validation']=='passed' and enum['layers']==layers
    assert sha(truth_from/'raw/truth_table.npz')==enum['truth_table_sha256']
    canon = json.dumps({'config':CONFIG,'layers':layers},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    config_hash = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+config_hash[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id
    for sub in ['raw/models','raw/samplers','logs']: (run_dir/sub).mkdir(parents=True)
    import platform, socket, importlib.metadata as md
    save(run_dir/'metadata.json', {'study':'mlfilter-scale','phase':'solver-fixedfm','protocol_phase':'exploratory','layers':layers,
         'config_hash':config_hash,'config_id':base.name,'config_hash_method':'sha256 of canonical JSON {config: solver_fixedfm.json, layers: L}',
         'run_id':run_id,'created_utc':now(),'code_sha256':code_hashes(),
         'truth_from':{'run_dir':str(truth_from.relative_to(PROJECT)),'truth_table_sha256':enum['truth_table_sha256']},
         'environment':{'python':sys.version,'platform':platform.platform(),'hostname':socket.gethostname(),
                        'packages':{n:md.version(n) for n in ['numpy','torch','dimod','dwave-neal','openqarp']}}})
    print(run_dir)


def load_meta(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    table = PROJECT/meta['truth_from']['run_dir']/'raw/truth_table.npz'
    assert sha(table)==meta['truth_from']['truth_table_sha256']
    return meta, np.load(table)['fom']


# ------------------------------------------------------------------ frozen FMs
def freeze(run_dir):
    import torch
    torch.set_num_threads(1)
    meta, truth = load_meta(run_dir); L = meta['layers']; x = M.onehot(M.labels_for(L))
    for seed in CONFIG['fm']['seeds']:
        out = run_dir/f'raw/models/seed{seed}.json'; assert not out.exists()
        ids = M.initial_ids(seed, L); ys = [float(truth[i]) for i in ids]
        m, qd, q, b, s = M.fit(x[ids], ys, seed); pred = M.predict_all(x, q, b)
        with torch.no_grad():
            np.testing.assert_allclose(pred[:4096], m(torch.tensor(x[:4096],dtype=torch.float32)).numpy(), atol=1e-4, rtol=0)
        cp = run_dir/f'raw/models/seed{seed}.pt'; torch.save(m.state_dict(), cp)
        save(out, {'seed':seed,'layers':L,'initial_ids':ids,'initial_values':ys,'model_sha256':M.model_hash(m),
                   'checkpoint':cp.name,'checkpoint_sha256':sha(cp),'S':s,'created_utc':now()})
        print('frozen FM seed', seed, flush=True)


def frozen(run_dir, seed, L):
    import torch
    from fm import TorchFM
    from fm_to_qubo import fm_to_qubo
    from qarp_backend import qubo_to_ising_coefficients
    d = json.loads((run_dir/f'raw/models/seed{seed}.json').read_text()); cp = run_dir/'raw/models'/d['checkpoint']
    assert sha(cp)==d['checkpoint_sha256']
    m = TorchFM(4*L, M.TRAINING['rank']); m.load_state_dict(torch.load(cp, weights_only=True)); m.eval()
    assert M.model_hash(m)==d['model_sha256']
    qd, b = fm_to_qubo(m); q = M.matrix(qd, 4*L)
    h, j = qubo_to_ising_coefficients(q); s = max([abs(v) for v in h.values()]+[abs(v) for v in j.values()]+[0.])
    assert abs(s-d['S']) < 1e-12
    x = M.onehot(M.labels_for(L)); pred = M.predict_all(x, q, b)
    return d, qd, q, b, s, x, pred


class Context:
    """Per-FM quantities shared by all samplers."""
    def __init__(self, pred, seen, L):
        self.pred, self.L, self.seen = pred, L, set(seen)
        mask = np.ones(len(pred), bool); mask[list(seen)] = False
        self.unseen_min, self.unseen_max = float(pred[mask].min()), float(pred[mask].max())
        order = np.argsort(pred, kind='stable'); pct = np.empty(len(pred)); pct[order] = np.arange(len(pred))/(len(pred)-1)
        self.pct = pct                                  # FM percentile over all candidates (0 = best)
        un = np.flatnonzero(mask); un_order = un[np.argsort(pred[un], kind='stable')]
        self.unseen_rank = np.full(len(pred), -1); self.unseen_rank[un_order] = np.arange(len(un))
        self.n_unseen = len(un)
        self.targets = un[np.abs(pred[un]-self.unseen_min) < 1e-8]

    def batch(self, raw, x_all):
        chosen, f = M.select(raw, self.pred, x_all, self.seen, self.L)
        ids = [i for i in M.bits_to_ids(raw, self.L) if i is not None]
        span = max(self.unseen_max-self.unseen_min, 1e-300)
        return {'selected_id':chosen, 'capture':bool(chosen is not None and abs(self.pred[chosen]-self.unseen_min)<1e-8),
                'norm_gap':1. if chosen is None else float((self.pred[chosen]-self.unseen_min)/span),
                'fm_rank_pct':1. if chosen is None else float(self.unseen_rank[chosen]/(self.n_unseen-1)),
                'feasible':f['raw_feasible_rate'], 'unique_unseen':f['unique_unseen_count'],
                'sample_fm_pct':float(np.mean(self.pct[ids])) if ids else None}

    def exact(self, prob):
        """prob over all candidates (feasible basis order = candidate index)."""
        mass = float(prob[self.targets].sum()); mass = min(1., max(0., mass))
        return {'P_unseen_min':mass, 'ideal_capture100':float(-np.expm1(100*np.log1p(-mass))) if mass<1 else 1.,
                'expected_fm_pct':float(prob@self.pct), 'seen_probability':float(prob[list(self.seen)].sum())}


def neighbor_fm_delta(pred, L, seed, n=4096):
    """Median |FM difference| between a candidate and its one-layer-change neighbours (feasible moves)."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, 4**L, n); labels = M.labels_for(L)[idx]
    diffs = []
    for g in range(L):
        w = 4**(L-1-g)
        for c in range(4):
            nb = idx + (c - labels[:, g].astype(np.int64))*w; keep = labels[:, g] != c
            diffs.append(np.abs(pred[nb[keep]]-pred[idx[keep]]))
    return float(np.median(np.concatenate(diffs)))


# ------------------------------------------------------------------ SA + RandomFM
def run_sa(run_dir, seed):
    meta, _ = load_meta(run_dir); L = meta['layers']; cfg = CONFIG['samplers']
    out = run_dir/f'raw/samplers/sa_seed{seed}.json'; assert not out.exists()
    d, qd, q, b, s, x, pred = frozen(run_dir, seed, L); ctx = Context(pred, d['initial_ids'], L)
    dE = neighbor_fm_delta(pred, L, seed)
    res = {'status':'running','seed':seed,'layers':L,'S':s,'neighbor_fm_delta_median':dE,'batches':[]}; save(out,res)
    try:
        for alpha, rep in itertools.product(cfg['SA']['alphas'], range(cfg['SA']['reps'])):
            t = time.perf_counter(); raw, sm = M.sample_sa(qd, b, s, float(alpha), L, seed*100000+rep)
            row = {'sampler':f'SA-a{alpha:g}','alpha':alpha,'rep':rep,'seconds':time.perf_counter()-t,**ctx.batch(raw,x),
                   'lambda':sm['lambda'],'beta_hot':sm['beta_range'][0],'beta_cold':sm['beta_range'][1],
                   'contrast_freeze':dE/sm['lambda'],'contrast_cold':sm['beta_range'][1]*dE,'barrier_cold':sm['beta_range'][1]*sm['lambda']}
            res['batches'].append(row)
        for rep in range(cfg['RandomFM']['reps']):
            raw, _ = M.sample_random_feasible(L, seed*100000+rep)
            res['batches'].append({'sampler':'RandomFM','rep':rep,**ctx.batch(raw,x)})
        res.update(status='completed', ended_utc=now()); save(out,res); print('SA/RandomFM seed', seed, 'done', flush=True)
    except Exception:
        res.update(status='failed', traceback=traceback.format_exc()); save(out,res); raise


# ------------------------------------------------------------------ XY-FMQAOA
class BudgetReached(Exception):
    pass


def embed(angles, old, new):
    a = list(angles); return a[:old]+[0.]*(new-old)+a[old:]+[0.]*(new-old)


def xy_search(backend, ctx, scale, seed, depth, objective, warm):
    from scipy.optimize import minimize
    from qarp_backend import OpenQARPCompactXYQAOA
    trace = []; best = None; bestprob = None
    rng = np.random.default_rng(np.random.SeedSequence([seed,0,1016,49,depth]))
    maxima = np.array([2*np.pi]*depth+[np.pi]*depth); bounds = [(0.,v) for v in maxima]
    def value(prob):
        return float(prob@ctx.pred) if objective=='mean' else M_cvar(ctx.pred, prob, .1)
    def evaluate(a, stop=1024, phase='global'):
        nonlocal best, bestprob
        if len(trace) >= stop: raise BudgetReached()
        if len(trace) % 32 == 0: OpenQARPCompactXYQAOA._mixer_cache.clear()
        a = np.asarray(a, float); prob = backend.feasible_probabilities(a[:depth], a[depth:]); v = value(prob)
        trace.append({'angles':a.tolist(),'phase':phase,'objective':v})
        if best is None or v < best['objective']: best, bestprob = trace[-1].copy(), prob.copy()
        return v/scale
    for g, bt in itertools.product(GRID['gammas'], GRID['betas']): evaluate(embed([g,bt],1,depth), phase='p1_grid49_embed')
    grid_best = (best.copy(), bestprob.copy()) if depth==1 else None
    if warm is not None: evaluate(embed(warm['angles'],depth-1,depth), phase='previous_depth_best_embed')
    while len(trace) < 256: evaluate(rng.random(2*depth)*maxima, phase='global_to256')
    for a in [r['angles'] for r in sorted(trace,key=lambda r:r['objective'])[:4]]:
        stop = len(trace)+192
        try: minimize(lambda v: evaluate(v,stop,'Powell'), a, method='Powell', bounds=bounds, options={'maxfev':192,'xtol':1e-6,'ftol':1e-9})
        except BudgetReached: pass
    while len(trace) < 1024: evaluate(rng.random(2*depth)*maxima, phase='remaining_global')
    assert len(trace)==1024
    if warm is not None: assert best['objective'] <= warm['objective']+1e-10
    return best, bestprob, grid_best, [r['objective'] for r in trace]


def M_cvar(energy, prob, alpha):
    # Same as intern_1009/perovskites/xy_cvar_sampling/py/experiment.py:cvar
    order = np.argsort(energy, kind='stable'); p = prob[order]/prob.sum(); before = np.cumsum(p)-p
    weights = np.maximum(0, np.minimum(p, alpha-before)); return float(weights@energy[order]/alpha)


def sample_from(prob, x, ctx, seed, reps, label, extra):
    rows = []; cdf = np.cumsum(prob)
    for rep in range(reps):
        draw = np.minimum(np.searchsorted(cdf, np.random.default_rng(seed*100000+rep).random(100)*cdf[-1], side='left'), len(prob)-1)
        rows.append({'sampler':label,'rep':rep,**extra,**ctx.batch(x[draw], x)})
    return rows


def run_xy(run_dir, seed, objective):
    from qarp_pair_xy import OpenQARPPairXYQAOA
    meta, _ = load_meta(run_dir); L = meta['layers']; cfg = CONFIG['samplers']['XY-FMQAOA']
    out = run_dir/f'raw/samplers/xy_{objective}_seed{seed}.json'; assert not out.exists()
    d, qd, q, b, s, x, pred = frozen(run_dir, seed, L); ctx = Context(pred, d['initial_ids'], L)
    res = {'status':'running','seed':seed,'layers':L,'objective':objective,'S':s,'searches':[],'batches':[]}; save(out,res)
    try:
        backend = OpenQARPPairXYQAOA(q/s, [4]*L); warm = None
        for depth in cfg['depths']:
            t = time.perf_counter(); best, prob, grid_best, objs = xy_search(backend, ctx, s, seed, depth, objective, warm)
            sec = time.perf_counter()-t
            assert abs(prob.sum()-1) < 1e-10
            res['searches'].append({'sampler':f'XY-p{depth}-{objective}','depth':depth,'budget':1024,'seconds':sec,'best':best,
                                    **ctx.exact(prob),'objective_trace_min_at':[int(np.argmin(objs[:k+1])) for k in (48,255,1023)]})
            res['batches'] += sample_from(prob, x, ctx, seed, CONFIG['samplers']['SA']['reps'], f'XY-p{depth}-{objective}', {'depth':depth})
            if grid_best is not None:
                gb, gprob = grid_best
                res['searches'].append({'sampler':f'XY-p1grid49-{objective}','depth':1,'budget':49,'best':gb,**ctx.exact(gprob)})
                res['batches'] += sample_from(gprob, x, ctx, seed, CONFIG['samplers']['SA']['reps'], f'XY-p1grid49-{objective}', {'depth':1})
            warm = best; save(out,res); print('XY', objective, 'seed', seed, 'p', depth, f'{sec:.0f}s', flush=True)
        res.update(status='completed', ended_utc=now()); save(out,res)
    except Exception:
        res.update(status='failed', traceback=traceback.format_exc()); save(out,res); raise


# ------------------------------------------------------------------ standard QAOA (X mixer + penalty)
def run_xstd(run_dir, seed, alpha):
    from fmqa_solver import build_penalty_bqm
    from qarp_backend import OpenQARPStandardQAOA
    from intern_0924.src.solvers import _build_qubo_energies, _state_energy_expectation, _sample_statevector_streaming
    meta, _ = load_meta(run_dir); L = meta['layers']; assert L in CONFIG['samplers']['Standard-QAOA']['layers']
    n = 4*L; out = run_dir/f'raw/samplers/xstd_a{alpha:g}_seed{seed}.json'; assert not out.exists()
    d, qd, q, b, s, x, pred = frozen(run_dir, seed, L); ctx = Context(pred, d['initial_ids'], L)
    res = {'status':'running','seed':seed,'layers':L,'alpha':alpha,'S':s,'trace':[],'batches':[]}; save(out,res)
    try:
        backend = OpenQARPStandardQAOA(q/s, [4]*L, alpha)
        total, offset = build_penalty_bqm({k:v/s for k,v in qd.items()}, [4]*L, alpha, b/s).to_qubo()
        energies = _build_qubo_energies(M.matrix(total, n), offset, np.dtype(np.float64))
        best = None; best_state = None; t0 = time.perf_counter()
        for g, bt in itertools.product(GRID['gammas'], GRID['betas']):
            state = backend.statevector([g],[bt]); v, norm = _state_energy_expectation(state, energies, 1<<16); assert abs(norm-1)<1e-10
            res['trace'].append({'gamma':g,'beta':bt,'objective':v})
            if best is None or v < best['objective']: best, best_state = res['trace'][-1].copy(), state
            else: del state
        valid = x.astype(np.uint64) @ (np.uint64(1) << np.arange(n, dtype=np.uint64))
        fprob = np.abs(best_state[valid])**2; feas_mass = float(fprob.sum())
        res['search'] = {'sampler':f'X-QAOA-a{alpha:g}','budget':49,'seconds':time.perf_counter()-t0,'best':best,'feasible_probability':feas_mass,
                         **ctx.exact(fprob/feas_mass if feas_mass>0 else fprob)}
        for rep in range(CONFIG['samplers']['SA']['reps']):
            idx = _sample_statevector_streaming(best_state, 100, seed*100000+rep, 1<<16)
            raw = ((idx[:,None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
            res['batches'].append({'sampler':f'X-QAOA-a{alpha:g}','alpha':alpha,'rep':rep,**ctx.batch(raw, x)})
        res.update(status='completed', ended_utc=now()); save(out,res); print('X-QAOA alpha', alpha, 'seed', seed, 'done', flush=True)
    except Exception:
        res.update(status='failed', traceback=traceback.format_exc()); save(out,res); raise


# ------------------------------------------------------------------ analysis
def analyze(run_dir):
    meta, _ = load_meta(run_dir); L = meta['layers']
    per_fm = {}; exact = {}; unfinished = []
    for f in sorted((run_dir/'raw/samplers').glob('*.json')):
        r = json.loads(f.read_text())
        if r['status']!='completed': unfinished.append(f.name); continue
        for row in r['batches']:
            per_fm.setdefault((row['sampler'], r['seed']), []).append(row)
        for srch in r.get('searches', []) + ([r['search']] if 'search' in r else []):
            exact.setdefault(srch['sampler'], []).append({'seed':r['seed'], **{k:srch[k] for k in srch if k in
                ('P_unseen_min','ideal_capture100','expected_fm_pct','seconds','feasible_probability')}})
    keys = ['capture','norm_gap','fm_rank_pct','feasible','sample_fm_pct','contrast_freeze','contrast_cold','barrier_cold','beta_hot','beta_cold','lambda']
    fm_means = {}
    for (sampler, seed), rows in per_fm.items():
        fm_means.setdefault(sampler, []).append({'seed':seed, **{k:float(np.mean([r[k] for r in rows if r.get(k) is not None]))
                                                for k in keys if any(r.get(k) is not None for r in rows)}})
    def q(v): v = np.asarray(v, float); return [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    summary = {s:{'n_fm':len(v), **{k:q([x[k] for x in v]) for k in keys if all(k in x for x in v)}} for s, v in fm_means.items()}
    exact_summary = {s:{'n_fm':len(v), **{k:q([x[k] for x in v]) for k in v[0] if k!='seed'}} for s, v in exact.items()}
    save(run_dir/'raw/summary.json', {'layers':L,'generated_utc':now(),'summary':summary,'exact':exact_summary,
                                      'per_fm':fm_means,'per_fm_exact':exact,'unfinished':unfinished})
    def order(s):
        if s.startswith('SA-a'): return (0, float(s[4:]))
        return {'RandomFM':(1,0)}.get(s, (2 if s.startswith('XY') else 3, s))
    print(f'== L={L} (median over FMs of 10-rep means)')
    for s in sorted(summary, key=order):
        v = summary[s]
        print(f"  {s:20s} capture {v['capture'][0]:.2f}  norm_gap {v['norm_gap'][0]:.3f}  rank_pct {v['fm_rank_pct'][0]:.4f}  feas {v['feasible'][0]:.2f}"
              + (f"  sample_pct {v['sample_fm_pct'][0]:.3f}" if 'sample_fm_pct' in v else '')
              + (f"  dE/lambda {v['contrast_freeze'][0]:.3g}  beta_hot {v['beta_hot'][0]:.3g}" if 'contrast_freeze' in v else ''))
    for s in sorted(exact_summary, key=order):
        v = exact_summary[s]
        print(f"  [exact] {s:20s} P_unseen_min {v['P_unseen_min'][0]:.4f}  ideal_capture100 {v['ideal_capture100'][0]:.3f}  E[fm_pct] {v['expected_fm_pct'][0]:.3f}")
    if unfinished: print('unfinished', unfinished)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','freeze','sa','xy','xstd','analyze'])
    ap.add_argument('--layers', type=int); ap.add_argument('--truth-from', type=Path); ap.add_argument('--run-dir', type=Path)
    ap.add_argument('--seed', type=int); ap.add_argument('--objective', choices=['mean','cvar10']); ap.add_argument('--alpha', type=float)
    a = ap.parse_args()
    import torch; torch.set_num_threads(1)
    if a.command=='init': init(a.layers, a.truth_from)
    elif a.command=='freeze': freeze(a.run_dir.resolve())
    elif a.command=='sa': run_sa(a.run_dir.resolve(), a.seed)
    elif a.command=='xy': run_xy(a.run_dir.resolve(), a.seed, a.objective)
    elif a.command=='xstd': run_xstd(a.run_dir.resolve(), a.seed, a.alpha)
    else: analyze(a.run_dir.resolve())
