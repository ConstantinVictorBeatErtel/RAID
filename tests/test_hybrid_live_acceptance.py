import hashlib
import json
import time
from pathlib import Path

import pytest
from raid_experiments import hybrid_live_acceptance as proof


def record(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    evidence = tmp_path / 'collection'
    files = {}
    for name in proof.REQUIRED_CODE:
        path = repo / name.removeprefix('repo/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# synthetic fixture: never live acceptance\n')
        files[name] = proof.sha(path)
    files['host/raid_experiments/managed_source.py'] = 'b' * 64
    library=repo.parent/'inputs/libero_repo/libero/envs/environment.py'
    library.parent.mkdir(parents=True,exist_ok=True)
    library.write_text('# pinned library fixture\n')
    files['inputs/libero_repo/libero/envs/environment.py']=proof.sha(library)
    protocol = {'digest': 'c' * 64, 'model_digest': 'd' * 64}
    original = repo / 'artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json'
    record(original, protocol)
    for name in ['requirements-wm-cosmos.lock', 'requirements-wm-sim.lock']:
        (repo / name).write_text('# fixture\n')
    runtime = hashlib.sha256((proof.IMAGE + proof.sha(repo / 'requirements-wm-cosmos.lock') + proof.sha(repo / 'requirements-wm-sim.lock')).encode()).hexdigest()
    bindings = {'code': hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), 'protocol': proof.sha(original), 'runtime': runtime}
    identity = {'package_sha256': 'a' * 64, 'bindings': bindings, 'protocol_digest': protocol['digest'], 'model_digest': protocol['model_digest']}
    record(evidence / 'launch_identity.json', identity)
    record(evidence / 'launch_code_binding_files.json', files)
    operational = {'acceptance_session_id': 'test-acceptance-id', 'service_deployment_sha256': hashlib.sha256(b'deployment').hexdigest(), 'expected_artifact_sha256': 'e' * 64,'source_code_sha256':bindings['code'],'package_sha256':identity['package_sha256'],'runtime_sha256':runtime}
    capsule = {'type': 'raid-v6-live-acceptance-v1', 'sessionId': operational['acceptance_session_id'], 'acceptedRelease': {'packageSha256': identity['package_sha256'], 'bindings': bindings, 'identitySha256': proof.sha(evidence / 'launch_identity.json')}, 'imageReference': proof.IMAGE, 'deployment': 'deployment', 'recoveredSha256': 'e' * 64, 'collectionSha256': 'e' * 64, 'terminationReason': 'independent_alarm', 'completedAt': int(time.time() * 1000)}
    monkeypatch.setattr(proof, 'fetch_evidence', lambda _: capsule)
    return repo, evidence, protocol, operational, capsule


def check(accepted):
    repo, evidence, protocol, operational, _ = accepted
    return proof.verify_live_acceptance(protocol, evidence, operational, repo_root=repo)


def test_exact_executing_release_requires_authoritative_correspondence(accepted):
    assert check(accepted)['scientific_result'] is False
    accepted[4]['acceptedRelease']['packageSha256'] = 'f' * 64
    with pytest.raises(ValueError, match='differs from intended launch'):
        check(accepted)


def test_changed_code_is_rejected_before_authority_request(accepted, monkeypatch):
    repo = accepted[0]
    (repo / 'raid_experiments/hybrid_runner.py').write_text('# changed\n')
    monkeypatch.setattr(proof, 'fetch_evidence', lambda _: pytest.fail('changed code must fail locally'))
    with pytest.raises(ValueError, match='executing code differs'):
        check(accepted)

def test_changed_libero_source_is_rejected_before_authority_request(accepted,monkeypatch):
    (accepted[0].parent/'inputs/libero_repo/libero/envs/environment.py').write_text('# changed\n')
    monkeypatch.setattr(proof,'fetch_evidence',lambda _:pytest.fail('changed library must fail locally'))
    with pytest.raises(ValueError,match='LIBERO code differs'):
        check(accepted)

@pytest.mark.parametrize('field',['source_code_sha256','package_sha256','runtime_sha256'])
def test_operational_record_bindings_must_match_actual_release(accepted,field):
    accepted[3][field]='f'*64
    with pytest.raises(ValueError,match='operational record differs'):
        check(accepted)


@pytest.mark.parametrize('field,value', [('collectionSha256', 'f' * 64), ('deployment', 'other'), ('completedAt', 1), ('terminationReason', 'operator_request')])
def test_authority_cannot_be_replaced_by_local_boolean_claims(accepted, field, value):
    accepted[3].update(status='passed', termination_verified=True, recovery_verified=True)
    accepted[4][field] = value
    with pytest.raises(ValueError):
        check(accepted)


def test_unbound_or_escaping_release_rejected(accepted):
    repo, evidence, *_ = accepted
    path = evidence / 'launch_code_binding_files.json'
    path.unlink()
    record(repo / 'elsewhere.json', {})
    path.symlink_to(repo / 'elsewhere.json')
    with pytest.raises(ValueError, match='escapes collection root'):
        check(accepted)


def test_invalid_locator_cannot_contact_network():
    with pytest.raises(ValueError, match='invalid live acceptance'):
        proof.fetch_evidence('../foreign-host')
