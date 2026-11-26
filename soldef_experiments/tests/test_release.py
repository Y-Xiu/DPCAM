import copy
import json
from collections import Counter, defaultdict
import numpy as np
import pytest
from common import ROOT
from metrics import ConfusionMatrix
from scripts.prepare_release import verify_membership, check_fresh
from comparison_study.report import metric_values


def test_published_source_splits_and_duplicate_protection():
    expected = {'soldef_task1': (159,34,35), 'soldef': (140,30,30)}
    for dataset, counts in expected.items():
        reference = json.loads((ROOT/f'data/splits/{dataset}.json').read_text())
        verify_membership(reference, reference)
        sources = defaultdict(set)
        for row in reference['records']:
            sources[row['source_sha256']].add(row['split'])
        assert all(len(v) == 1 for v in sources.values())
        actual = Counter(next(iter(v)) for v in sources.values())
        assert tuple(actual[s] for s in ('train','val','test')) == counts
        changed = copy.deepcopy(reference)
        changed['records'][0]['split'] = 'test' if changed['records'][0]['split'] != 'test' else 'train'
        with pytest.raises(ValueError, match='membership'): verify_membership(changed, reference)
        duplicate = copy.deepcopy(reference);duplicate['records'].append(duplicate['records'][0])
        with pytest.raises(ValueError, match='Duplicate'): verify_membership(duplicate, reference)


def test_configuration_preparation_cannot_invalidate_existing_run(tmp_path):
    check_fresh(tmp_path)
    folder=tmp_path/'two_task_runs/a/seed_42';folder.mkdir(parents=True)
    (folder/'status.json').write_text('{}')
    with pytest.raises(RuntimeError, match='fresh checkout'): check_fresh(tmp_path)


def test_corrupt_validation_is_excluded_and_valid_metrics_keep_units(tmp_path, monkeypatch):
    import two_task_study.report as report
    monkeypatch.setattr(report,'ROOT',tmp_path)
    jobs=[{'task':'task1','variant':'dpcam','seed':s} for s in (42,43)]
    names=['background','good','no_good']
    cm=ConfusionMatrix(3);cm.matrix=np.array([[8,1,1],[1,7,2],[2,1,7]])
    value={'split':'val','reloaded_checkpoint_verified':True,**cm.compute(names)}
    for job in jobs:
        run=report.run_dir(job);run.mkdir(parents=True)
        (run/'status.json').write_text(json.dumps({'status':'completed','epoch':150,'evaluation_completed':True,'validation_reload_verified':True}))
        (run/'validation_metrics.json').write_text(json.dumps(value) if job['seed']==42 else '')
    rows, errors=report.collect({'jobs':jobs})
    assert len(errors)==1 and sum('metrics' in r for r in rows)==1
    assert rows[0]['metrics']['Recall_defect_percent']==pytest.approx(70)
    assert rows[0]['metrics']['F1_macro']==pytest.approx(value['mDice'])
    value['mIoU']=float('nan')
    with pytest.raises(ValueError):metric_values(value,'task1')


def test_final_selection_rejects_invalid_artifacts_before_test(tmp_path,monkeypatch):
    import comparison_study.finalize as final
    monkeypatch.setattr(final,'ROOT',tmp_path)
    jobs=[{'run':f'runs/{i}','task':'task1'} for i in range(150)]
    monkeypatch.setattr(final,'all_jobs',lambda:jobs)
    with pytest.raises(ValueError,match='incomplete selection'):final.freeze_selection()
    assert not (tmp_path/'reports/comparison_study/selection_lock.json').exists()
