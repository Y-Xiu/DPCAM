"""Exclusive, resumable two-task queue (task1 placement and task2 solder state)."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, digest, write_json
from soldef_study.train import runtime_hashes, check_sources

PLAN_PATH = ROOT / 'configs/two_task/plan.json'
FOLDER = ROOT / 'reports/two_task'
RUNS = ROOT / 'two_task_runs'


def run_dir(entry):
    return RUNS / f'twotask_{entry["task"]}_{entry["variant"]}' / f'seed_{entry["seed"]}'


def report(plan):
    from two_task_study.report import generate
    generate()


def selected(plan, args):
    jobs = plan['jobs']
    if args.task != 'both':
        jobs = [j for j in jobs if j['task'] == args.task]
    if args.variants:
        keep = {v.strip() for v in args.variants.split(',') if v.strip()}
        unknown = keep - set(plan['variants'])
        if unknown:
            raise ValueError(f'Unknown variants: {sorted(unknown)}')
        jobs = [j for j in jobs if j['variant'] in keep]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    return jobs


def execute():
    p = argparse.ArgumentParser()
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--task', choices=['both', 'task1', 'task2'], default='both')
    p.add_argument('--variants', default='')
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    plan = json.loads(PLAN_PATH.read_text())
    assert plan['protocol_id'] == 'two_task_v1'
    if args.report_only:
        report(plan)
        return
    jobs = selected(plan, args)
    if not jobs:
        raise ValueError('No jobs selected')
    FOLDER.mkdir(parents=True, exist_ok=True)
    with (FOLDER / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_path = FOLDER / 'runtime_lock.json'
        if lock_path.exists():
            frozen = json.loads(lock_path.read_text())
            assert frozen['plan_sha256'] == digest(PLAN_PATH)
            check_sources(frozen['sources'])
        else:
            frozen = {'plan_sha256': digest(PLAN_PATH), 'sources': runtime_hashes()}
            write_json(lock_path, frozen)
        for entry in plan['jobs']:
            assert digest(ROOT / entry['config']) == entry['config_sha256']
        state = {'pid': os.getpid(), 'status': 'running', 'planned': len(jobs),
                 'failures': [], 'tasks': sorted({j['task'] for j in jobs}),
                 'test_split_accessed': False}
        write_json(FOLDER / 'queue_status.json', state)
        report(plan)
        for entry in jobs:
            check_sources(frozen['sources'])
            assert digest(ROOT / entry['config']) == entry['config_sha256']
            cfg = json.loads((ROOT / entry['config']).read_text())
            experiment = cfg['experiment']
            run = run_dir(entry)
            status = json.loads((run / 'status.json').read_text()) if (run / 'status.json').exists() else {}
            if status.get('evaluation_completed'):
                from comparison_study.suite import validated_run
                valid, reason = validated_run(ROOT, run, entry)
                if not valid:
                    raise RuntimeError('Completed run has invalid artifacts: ' + reason)
                if (run / 'last.pt').exists():
                    (run / 'last.pt').unlink()
                continue
            if shutil.disk_usage(ROOT).free < 10 * 1024 ** 3:
                state.update(status='blocked_disk_space')
                write_json(FOLDER / 'queue_status.json', state)
                return
            state.update(current=experiment, seed=entry['seed'])
            write_json(FOLDER / 'queue_status.json', state)
            command = [sys.executable, '-m', 'soldef_study.train']
            params = ['--config', entry['config'], '--seed', str(entry['seed'])]
            try:
                with (FOLDER / f"{experiment}_{entry['seed']}.log").open('a') as log:
                    if status.get('status') != 'completed':
                        resume = ['--resume'] if (run / 'last.pt').exists() else []
                        if run.exists() and not resume:
                            raise RuntimeError('Existing run without a resume checkpoint')
                        subprocess.run(command + ['train'] + params + resume, cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                    subprocess.run(command + ['evaluate'] + params, cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                if (run / 'last.pt').exists():
                    (run / 'last.pt').unlink()
            except Exception as exc:
                state['failures'].append({'experiment': experiment,
                                          'seed': entry['seed'], 'error': repr(exc)})
                print('FAILED', experiment, entry['seed'], repr(exc), flush=True)
                if not (run / 'history.json').exists():
                    state.update(status='failed_startup')
                    report(plan)
                    write_json(FOLDER / 'queue_status.json', state)
                    return
            report(plan)
            write_json(FOLDER / 'queue_status.json', state)
        state.update(status='completed' if not state['failures'] else 'completed_with_failures')
        report(plan)
        write_json(FOLDER / 'queue_status.json', state)


def main():
    try:
        execute()
    except BlockingIOError:
        raise SystemExit('Another two-task queue is active; no new worker launched.')
    except Exception as exc:
        path = FOLDER / 'queue_status.json'
        state = json.loads(path.read_text()) if path.exists() else {'pid': os.getpid()}
        if state.get('pid') == os.getpid():
            state.update(status='failed', error=repr(exc))
            write_json(path, state)
        raise


if __name__ == '__main__':
    main()
