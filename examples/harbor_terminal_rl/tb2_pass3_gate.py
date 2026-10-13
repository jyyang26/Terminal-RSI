#!/usr/bin/env python3
"""Validate a completed Harbor TB2 pass@3 result before training is allowed."""
from __future__ import annotations
import argparse,json,sys

def main():
 p=argparse.ArgumentParser(); p.add_argument('metrics_json'); p.add_argument('--min',type=float,default=.35); p.add_argument('--max',type=float,default=.45); p.add_argument('--tasks',type=int,default=89); p.add_argument('--k',type=int,default=3); a=p.parse_args()
 m=json.load(open(a.metrics_json))
 expected=a.tasks*a.k
 errors=[]
 def eq(key,val):
  if m.get(key)!=val: errors.append(f'{key}={m.get(key)!r}, expected {val!r}')
 eq('harbor/eval_pass_at_k',a.k); eq('harbor/eval_num_tasks',a.tasks); eq('harbor/eval_num_samples',expected); eq('harbor/eval_num_missing_attempts',0)
 coverage=float(m.get('harbor/eval_attempt_coverage',0)); rate=float(m.get('harbor/eval_resolved_rate_full_pass@3',m.get('harbor/eval_resolved_rate_full',-1))); p1=float(m.get('harbor/eval_resolved_rate_full_pass@1',m.get('harbor/eval_pass_at_1_first_attempt_full',-1))); avg3=float(m.get('harbor/eval_resolved_rate_full_avg@3',-1))
 if coverage < .999999: errors.append(f'attempt coverage={coverage}, expected 1.0')
 if rate + 1e-12 < p1: errors.append(f'pass@3={rate} < first-attempt pass@1={p1}')
 if avg3 < 0 or avg3 > rate + 1e-12: errors.append(f'avg@3={avg3} must satisfy 0 <= avg@3 <= pass@3={rate}')
 if not (a.min <= rate <= a.max): errors.append(f'pass@3={rate:.6f}, expected [{a.min},{a.max}]')
 if errors:
  print('TB2 pass@3 gate FAILED:'); print('\n'.join(' - '+x for x in errors)); return 1
 print(f'TB2 pass@3 gate PASSED: pass@3={rate:.6f}, avg@3={avg3:.6f}, pass@1={p1:.6f}, attempts={expected}/{expected}')
 return 0
if __name__=='__main__': raise SystemExit(main())
