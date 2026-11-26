"""Aggregate completed validation records, with task-aware units and artifact errors."""
import json
import math
import statistics
from common import ROOT, write_json
from comparison_study.report import LABELS, metric_values, read_artifact

ORDER = ('none', 'cbam', 'dpcam', 'no_dca', 'no_pa', 'no_dsa',
         'equal_spatial', 'residual_cbam', 'residual_dpcam')
TASKS = ('task1', 'task2')


def run_dir(entry):
    return ROOT / 'two_task_runs' / f'twotask_{entry["task"]}_{entry["variant"]}' / f'seed_{entry["seed"]}'


def collect(plan):
    rows, errors = [], []
    for entry in plan['jobs']:
        run = run_dir(entry)
        path = run / 'status.json'
        status = (read_artifact(path, errors) or {'status': 'invalid'}) if path.exists() else {'status': 'pending'}
        row = {**entry, 'status': status}
        if (status.get('status') == 'completed' and status.get('epoch') == 150
                and status.get('evaluation_completed') and status.get('validation_reload_verified')):
            path = run / 'validation_metrics.json'
            value = read_artifact(path, errors)
            if value is not None:
                try:
                    if value.get('split') != 'val' or not value.get('reloaded_checkpoint_verified'):
                        raise ValueError('Validation reload was not verified')
                    row['metrics'] = metric_values(value, entry['task'])
                    row['validation'] = value
                except (ValueError, KeyError, TypeError, AssertionError) as error:
                    errors.append({'file': str(path), 'error': repr(error)})
        rows.append(row)
    return rows, errors


def stats(values):
    return {'n': len(values), 'mean': statistics.mean(values),
            'sample_std': statistics.stdev(values) if len(values) > 1 else None}


def paired(rows, task, variant, baseline):
    def values(name):
        return {r['seed']: r['metrics']['mIoU_percent'] for r in rows
                if r['task'] == task and r['variant'] == name and 'metrics' in r}
    a, b = values(variant), values(baseline)
    seeds = sorted(set(a) & set(b))
    differences = [a[s] - b[s] for s in seeds]
    out = {'seeds': seeds, 'differences_percentage_points': differences,
           'complete_five_seeds': seeds == list(range(42, 47))}
    if differences:
        out.update(stats(differences))
        if len(differences) == 5:
            half = 2.7764451051977987 * out['sample_std'] / math.sqrt(5)
            out['nominal_paired_t_95CI_pp'] = [out['mean'] - half, out['mean'] + half]
    return out


def generate():
    folder = ROOT / 'reports/two_task'
    folder.mkdir(parents=True, exist_ok=True)
    plan = json.loads((ROOT / 'configs/two_task/plan.json').read_text())
    rows, errors = collect(plan)
    validated = sum('validation' in r for r in rows)
    write_json(folder / 'results.json', {'planned': len(rows), 'validated': validated,
               'test_split_accessed': False, 'runs': rows, 'invalid_artifacts': errors})
    summary = {}
    lines = ['# SolDef_AI 两任务验证集汇总', '',
             f'可读取且标记重载验证通过的记录：{validated}/{len(rows)}；未完成项以 — 表示。',
             '本报告只读取已保存的验证指标，不进行模型推理，不读取测试图像。', '',
             'mIoU、mPA 按全部类别宏平均（含背景）；Recall_defect 在 Task 1 为 no_good，'
             '在 Task 2 为 exc_solder / poor_solder / spike 的宏平均召回。F1_macro 范围为 0–1。'
             '所有指标为实际完成种子的均值 ± 样本标准差（ddof=1）。', '',
             'Task 1：159/34/35 原图，318/68/70 裁剪；Task 2：140/30/30 原图，280/60/60 裁剪。'
             '同一原图不跨集合；输入 512，150 轮，五训练种子 42–46。', '']
    for task in TASKS:
        summary[task] = {}
        lines += [f'## {task}', '', '| Model | n | mIoU (%) | mPA (%) | Recall_defect (%) | F1_macro |',
                  '|---|---:|---:|---:|---:|---:|']
        for variant in ORDER:
            values = [r['metrics'] for r in rows if r['task'] == task and r['variant'] == variant and 'metrics' in r]
            item = {k: stats([v[k] for v in values]) for k in values[0]} if values else {}
            summary[task][variant] = {'n': len(values), 'accuracy': item}
            def fmt(key):
                if key not in item:
                    return '—'
                v = item[key]
                return (f"{v['mean']:.3f} ± {v['sample_std']:.3f}" if v['sample_std'] is not None
                        else f"{v['mean']:.3f} (n=1)")
            cells = [LABELS[variant], str(len(values))] + [fmt(k) for k in (
                'mIoU_percent', 'mPA_percent', 'Recall_defect_percent', 'F1_macro')]
            lines.append('| ' + ' | '.join(cells) + ' |')
        summary[task]['paired'] = {f'{a}_vs_{b}': paired(rows, task, a, b) for a, b in (
            ('dpcam', 'none'), ('dpcam', 'cbam'), ('residual_dpcam', 'residual_cbam'))}
        lines.append('')
    lines += ['配对差值与名义 t 区间保存在 summary.json；小样本验证结果不能代替独立测试结论。',
              '复杂度与速度使用 comparison_study 的独立统一测量报告，未测量不填数值。']
    if errors:
        lines += ['', '## 无效 / 不完整文件', ''] + ['- ' + e['file'] + ': ' + e['error'] for e in errors]
    write_json(folder / 'summary.json', {'scope': 'two_task_validation_only', 'planned': len(rows),
               'validated': validated, 'tasks': summary, 'invalid_artifacts': errors})
    (folder / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    generate()
