"""XY-FMQAOA p=4..6 on the frozen FMs of a solver-fixedfm run, continuing from its p=3 best angles (exploratory).

  init    --solver-run D                   new run dir linked to a finished solver-fixedfm run
  run     --run-dir R --seed S --objective mean|cvar10
  analyze --run-dir R                      summary.json in the same schema as solver-fixedfm
"""
import argparse
import hashlib
import json
import secrets
import sys
import time
import traceback
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import solver_fixedfm as SF  # noqa: E402  (sets sys.path for src/ and Intern_Fujitsu)

STUDY_DIR, PROJECT = SF.STUDY_DIR, SF.PROJECT
CONFIG = json.loads((STUDY_DIR/'xy_deep.json').read_text())
CODE = [Path(__file__).resolve(), STUDY_DIR/'xy_deep.json', *SF.CODE]


def code_hashes():
    return {str(p): SF.sha(p) for p in CODE}


def init(solver_run):
    solver_run = solver_run.resolve(); smeta, _ = SF.load_meta(solver_run)
    inputs = {}
    for seed in SF.CONFIG['fm']['seeds']:
        for obj in CONFIG['objectives']:
            f = solver_run/f'raw/samplers/xy_{obj}_seed{seed}.json'; d = json.loads(f.read_text()); assert d['status']=='completed'
            inputs[f.name] = SF.sha(f)
    canon = json.dumps({'config':CONFIG,'layers':smeta['layers'],'solver_config_hash':smeta['config_hash']},sort_keys=True,ensure_ascii=False,separators=(',',':'))
    config_hash = hashlib.sha256(canon.encode()).hexdigest(); base = PROJECT/'data/runs/mlfilter-scale'/('c-'+config_hash[:12])
    while True:
        run_id = 'r-'+secrets.token_hex(6)
        if not (base/run_id).exists(): break
    run_dir = base/run_id
    for sub in ['raw/samplers','logs']: (run_dir/sub).mkdir(parents=True)
    SF.save(run_dir/'metadata.json', {'study':'mlfilter-scale','phase':'solver-xydeep','protocol_phase':'exploratory','layers':smeta['layers'],
            'config_hash':config_hash,'config_id':base.name,'config_hash_method':'sha256 of canonical JSON {config: xy_deep.json, layers, solver_config_hash}',
            'run_id':run_id,'created_utc':SF.now(),'code_sha256':code_hashes(),'solver_run':str(solver_run.relative_to(PROJECT)),
            'solver_inputs_sha256':inputs,'environment':smeta['environment']})
    print(run_dir)


def load(run_dir):
    meta = json.loads((run_dir/'metadata.json').read_text())
    assert meta['code_sha256']==code_hashes(), 'code changed since init; create a new run'
    solver_run = PROJECT/meta['solver_run']; SF.load_meta(solver_run)
    return meta, solver_run


def run(run_dir, seed, objective):
    from qarp_pair_xy import OpenQARPPairXYQAOA
    meta, solver_run = load(run_dir); L = meta['layers']
    src = solver_run/f'raw/samplers/xy_{objective}_seed{seed}.json'; assert SF.sha(src)==meta['solver_inputs_sha256'][src.name]
    base = json.loads(src.read_text()); warm = next(s['best'] for s in base['searches'] if s['sampler']==f'XY-p3-{objective}')
    out = run_dir/f'raw/samplers/xy_{objective}_seed{seed}.json'; assert not out.exists()
    d, qd, q, b, s, x, pred = SF.frozen(solver_run, seed, L); ctx = SF.Context(pred, d['initial_ids'], L)
    res = {'status':'running','seed':seed,'layers':L,'objective':objective,'S':s,'warm_start':{'from':str(src.relative_to(PROJECT)),'p3_best':warm},
           'searches':[],'batches':[]}; SF.save(out, res)
    try:
        backend = OpenQARPPairXYQAOA(q/s, [4]*L)
        for depth in CONFIG['depths']:
            t = time.perf_counter(); best, prob, _, objs = SF.xy_search(backend, ctx, s, seed, depth, objective, warm); sec = time.perf_counter()-t
            assert abs(prob.sum()-1) < 1e-10
            res['searches'].append({'sampler':f'XY-p{depth}-{objective}','depth':depth,'budget':1024,'seconds':sec,'best':best,**ctx.exact(prob),
                                    'objective_trace_min_at':[int(np.argmin(objs[:k+1])) for k in (48,255,1023)]})
            res['batches'] += SF.sample_from(prob, x, ctx, seed, SF.CONFIG['samplers']['SA']['reps'], f'XY-p{depth}-{objective}', {'depth':depth})
            warm = best; SF.save(out, res); print('XY', objective, 'seed', seed, 'p', depth, f'{sec:.0f}s', flush=True)
        res.update(status='completed', ended_utc=SF.now()); SF.save(out, res)
    except Exception:
        res.update(status='failed', traceback=traceback.format_exc()); SF.save(out, res); raise


def analyze(run_dir):
    # Same aggregation as solver_fixedfm.analyze (10-rep mean per FM, median/IQR over FMs).
    meta, _ = load(run_dir); L = meta['layers']; per_fm = {}; exact = {}; unfinished = []
    for f in sorted((run_dir/'raw/samplers').glob('*.json')):
        r = json.loads(f.read_text())
        if r['status']!='completed': unfinished.append(f.name); continue
        for row in r['batches']: per_fm.setdefault((row['sampler'], r['seed']), []).append(row)
        for srch in r['searches']:
            exact.setdefault(srch['sampler'], []).append({'seed':r['seed'], **{k:srch[k] for k in ('P_unseen_min','ideal_capture100','expected_fm_pct','seconds')}})
    keys = ['capture','norm_gap','fm_rank_pct','feasible','sample_fm_pct']
    fm_means = {}
    for (sampler, seed), rows in per_fm.items():
        fm_means.setdefault(sampler, []).append({'seed':seed, **{k:float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys}})
    def q(v): v = np.asarray(v, float); return [float(np.median(v)), float(np.percentile(v,25)), float(np.percentile(v,75))]
    summary = {s:{'n_fm':len(v), **{k:q([x[k] for x in v]) for k in keys}} for s, v in fm_means.items()}
    exact_summary = {s:{'n_fm':len(v), **{k:q([x[k] for x in v]) for k in v[0] if k!='seed'}} for s, v in exact.items()}
    SF.save(run_dir/'raw/summary.json', {'layers':L,'generated_utc':SF.now(),'summary':summary,'exact':exact_summary,
                                         'per_fm':fm_means,'per_fm_exact':exact,'unfinished':unfinished})
    print(f'== L={L} XY deep (median over FMs)')
    for s in sorted(summary):
        v = summary[s]; e = exact_summary.get(s, {})
        print(f"  {s:16s} capture {v['capture'][0]:.2f}  norm_gap {v['norm_gap'][0]:.3f}  sample_pct {v['sample_fm_pct'][0]:.3f}"
              + (f"  ideal_capture100 {e['ideal_capture100'][0]:.3f}  sec {e['seconds'][0]:.0f}" if e else ''))
    if unfinished: print('unfinished', unfinished)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['init','run','analyze']); ap.add_argument('--solver-run', type=Path)
    ap.add_argument('--run-dir', type=Path); ap.add_argument('--seed', type=int); ap.add_argument('--objective', choices=['mean','cvar10'])
    a = ap.parse_args()
    import torch; torch.set_num_threads(1)
    if a.command=='init': init(a.solver_run)
    elif a.command=='run': run(a.run_dir.resolve(), a.seed, a.objective)
    else: analyze(a.run_dir.resolve())
