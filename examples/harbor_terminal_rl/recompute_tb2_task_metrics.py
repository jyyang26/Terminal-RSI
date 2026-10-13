#!/usr/bin/env python3
"""Recompute honest TB2 metrics from one verifier result per trial/task.

This intentionally ignores converted rollout Samples: trajectory_retokenized may
emit many chunks for one trial and must never change benchmark denominators.
"""
import argparse, glob, json, pathlib

p=argparse.ArgumentParser(); p.add_argument('trials_dir'); p.add_argument('--expected-tasks',type=int,default=89); p.add_argument('--output')
a=p.parse_args()
rows=[]
for f in sorted(glob.glob(str(pathlib.Path(a.trials_dir)/'*'/'result.json'))):
    try: d=json.load(open(f))
    except Exception: continue
    r=(d.get('verifier_result') or {}).get('rewards',{}).get('reward')
    if not isinstance(r,(int,float)): continue
    rows.append({'task_name':d.get('task_name'),'trial_name':d.get('trial_name'),'reward':float(r),'result_json':f})
resolved=sum(x['reward']>=0.5 for x in rows)
out={
 'expected_tasks':a.expected_tasks,
 'num_scored_tasks':len(rows),
 'num_unscored_tasks':a.expected_tasks-len(rows),
 'num_resolved_tasks':resolved,
 'num_unresolved_scored_tasks':len(rows)-resolved,
 'eval_reward_mean_task_level':sum(x['reward'] for x in rows)/len(rows) if rows else None,
 'eval_resolved_rate_full_task_level':resolved/a.expected_tasks if a.expected_tasks else None,
 'invariant_rate_between_0_and_1': 0 <= resolved/a.expected_tasks <= 1 if a.expected_tasks else False,
 'rows':rows,
}
text=json.dumps(out,indent=2,ensure_ascii=False)
print(text)
if a.output: pathlib.Path(a.output).write_text(text+'\n')
