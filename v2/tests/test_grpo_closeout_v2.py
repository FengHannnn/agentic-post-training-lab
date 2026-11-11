"""Keep the closeout intervention confined to the two maintenance slots."""
import json
from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from apt_v2 import grpo_closeout_study as closeout, grpo_lr_midpoint as midpoint
from apt_v2.grpo_warmup_probe import make_plan
from grpo_common import atomic_json, read, validate_config
from test_grpo_lr_midpoint_v2 import episode
from test_grpo_repair_study_v2 import result


def master():
    return read(ROOT/'configs/grpo-sql-closeout-v1.json')


def test_frozen_replacements_preserve_train_slots_and_r16_exposures():
    from grpo_repair_worker import sft_order, supervision_for, verify_for
    cfg = master()
    warm = closeout.warm_config(cfg, 4e-6)
    assert warm['sft'] == dict(midpoint.recipe(4e-6), supervision_sha256=closeout.DATASET_SHA)
    original = read(ROOT/cfg['probe_examples'])
    data = read(ROOT/cfg['sft_examples'])
    assert data == supervision_for(warm)
    assert verify_for(warm, data)
    assert len(data) == len(original) == 60
    changed = Counter(a['state'] for a, b in zip(original, data) if a != b)
    assert changed == {'failed_check_repair': 12, 'checked_submit': 12}
    batches = sft_order(warm)
    assert batches == read(ROOT/midpoint.INPUT_DIR/'order-r16.json')
    occurrences = Counter(i for batch in batches for i in batch)
    assert all(occurrences[i] == (16 if e['role'] == 'target' else 1) for i, e in enumerate(data))
    assert make_plan(original) == read(ROOT/cfg['input_dir']/'probe-plan.json')


def test_new_check_replaces_only_one_evidence_id_and_submit_preserves_checked_payload():
    data = read(ROOT/master()['sft_examples'])
    pairs = {(e['task_id'], e['state']): e for e in data}
    observed_ids = set()
    replaced_metrics = Counter()
    for (task, state), checked in pairs.items():
        if state != 'failed_check_repair':
            continue
        submitted = pairs[task, 'checked_submit']
        bad = json.loads(checked['messages'][-2]['content'])
        good = json.loads(checked['response'])
        assert bad['tool'] == good['tool'] == 'check_result'
        assert bad['arguments']['values'] == good['arguments']['values']
        old_ids, new_ids = (a['arguments']['evidence'] for a in (bad, good))
        changed = [k for k in old_ids if old_ids[k] != new_ids[k]]
        assert len(changed) == 1
        metric = changed[0]
        replaced_metrics[metric] += 1
        observed_ids.add((old_ids[metric], new_ids[metric]))
        assert submitted['messages'][:-2] == checked['messages']
        assert json.loads(submitted['messages'][-2]['content']) == good
        assert json.loads(submitted['response']) == {'tool': 'submit', 'arguments': good['arguments']}
        assert submitted['observation']['submitted']
    assert replaced_metrics == {'current_net_cents': 6, 'prior_net_cents': 6}
    assert len(observed_ids) > 1


def test_remaining_budget_and_prior_midpoint_proof_are_bound():
    cfg = master()
    closeout.validate_master(cfg)
    prior = read(ROOT/cfg['input_dir']/'prior-evidence.json')
    closeout.validate_prior(prior, cfg)
    assert cfg['max_wall_seconds'] == 9000 and cfg['prior_elapsed_seconds'] + 9000 < cfg['total_authorized_seconds']
    with pytest.raises(ValueError, match='frozen closeout'):
        closeout.validate_master({**cfg, 'max_wall_seconds': 9200})
    prior['receipts'][-1]['sha256'] = '0' * 64
    with pytest.raises(ValueError, match='ledger'):
        closeout.validate_prior(prior, cfg)


def test_new_branch_binds_candidate_identity_and_original_recipe_is_distinct(tmp_path, monkeypatch):
    import run_v2_grpo_dose as runner
    cfg = master()
    atomic_json(tmp_path/cfg['branch_template'], read(ROOT/cfg['branch_template']))
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    out = tmp_path/'artifacts'/cfg['run_id']
    assert runner.branch_config(cfg, out, 'w')['source_sft_updates'] == 513
    adapter = out/'warmups/closeout/warm-start/B/adapter/adapter_model.safetensors'
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b'closeout')
    branch = runner.branch_config(cfg, out, 'closeout')
    assert branch['source_sft_updates'] == 573 and branch['warm_learning_rate'] == 4e-6
    with pytest.raises(ValueError, match='candidate identity'):
        validate_config({**branch, 'run_id': cfg['run_id']+'-lr4'})


@pytest.mark.parametrize('passes', [False, True])
def test_closeout_runner_uses_the_frozen_gate_and_candidate_diagnostic_key(tmp_path, monkeypatch, passes):
    import grpo_dose_worker as worker
    import run_v2_grpo_dose as runner
    cfg = master()
    study = runner.Study(cfg, tmp_path)
    def child(name):
        path = tmp_path/'branches'/name
        atomic_json(path/'config.json', read(ROOT/cfg['branch_template']))
        return path
    def stage(label, *args):
        if label == 'gate-closeout':
            atomic_json(tmp_path/'train-gates/closeout.json', {'passed': True})
    def evaluate(child, name, *args):
        if name == 'w':
            return dict(result(first=0), elapsed_seconds=1800, episodes=[episode(True)])
        return dict(result(full=14 if passes else 13, target=8, first=9), episodes=[episode(passes)])
    monkeypatch.setattr(study, 'child', child)
    monkeypatch.setattr(study, 'stage', stage)
    monkeypatch.setattr(study, 'evaluate', evaluate)
    monkeypatch.setattr(worker, 'candidate_source', lambda *a: {'adapter_sha256': 'a'*64, 'source_sft_updates': 573})
    choice = study.execute()
    assert choice['selected_warm_start'] == ('closeout' if passes else None)
    assert (tmp_path/'ready-B.json').exists() == passes
    diag = read(tmp_path/'completion-diagnostics.json')
    assert set(diag['standard']['sample_recovery']) == {'w', 'closeout'}
