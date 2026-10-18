"""Portable four-case diagnostic on the existing independent Runpod lifecycle."""
from __future__ import annotations
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import tarfile
import time
import uuid
from . import operational_acceptance as ops,managed_source as managed

TASKS=('open_the_middle_drawer_of_the_cabinet','put_the_bowl_on_the_stove',
       'put_the_wine_bottle_on_the_rack','turn_on_the_stove')
MODULES=('__init__.py','wm_protocol.py','world_model.py','hybrid_protocol.py',
         'hybrid_generator_profile.py','cosmos_diagnostic.py','source_worker.py')
PARENT=ops.ROOT/'artifacts/raid_hybrid_libero_goal_v6/hf_launcher_fix_20261012/collection'
PROFILE=ops.ROOT/'artifacts/raid_hybrid_libero_goal_v6/cosmos_configuration_audit_20261013/protocol_v6_nvidia_square480.json'


def preflight(root,archive_hash):
    # Validate exact membership, sizes and bytes before importing packaged code.
    return "python3.12 - <<'PY'\n"+f'''
import hashlib,json,pathlib,shutil,tarfile,sys
r=pathlib.Path({root!r})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert sys.version_info[:2]==(3,12)
archive=r/'source_panel_package.tar'
assert sha(archive)=={archive_hash!r},'archive changed'
with tarfile.open(archive,'r:') as t:
 members=t.getmembers()
 assert all(m.isfile() and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts for m in members),'unsafe archive'
 assert len({{m.name for m in members}})==len(members),'duplicate archive member'
 manifest=json.load(t.extractfile('package_manifest.json'))
 rows=manifest['files'];expected={{x['path']:x for x in rows}}
 assert len(expected)==len(rows)
 assert set(expected)|{{'package_manifest.json'}}=={{m.name for m in members}},'unexpected member'
 assert all(m.name=='package_manifest.json' or m.size==expected[m.name]['size_bytes'] for m in members)
 assert shutil.disk_usage(r).free>=sum(m.size for m in members)*1.2+30*1024**3
 t.extractall(r,filter='data')
for x in rows:
 p=r/x['path'];assert p.stat().st_size==x['size_bytes'] and sha(p)==x['sha256'],'corrupt file: '+x['path']
sys.path.insert(0,str(r/'repo'))
from raid_experiments.cosmos_diagnostic import verified_cases
verified_cases(json.loads((r/'repo/protocol.json').read_text()),json.loads((r/'inputs/cases.json').read_text()),r/'inputs')
print('FOUR_SOURCE_INPUTS_VERIFIED',flush=True)
archive.unlink()
PY
'''


def prepare(session):
    import numpy as np
    import h5py
    from .world_model import resize_rgb,to_display
    from .hybrid_protocol import validate_hybrid_protocol
    session=Path(session);session.mkdir(parents=True,exist_ok=True)
    if (session/'release.json').exists():raise RuntimeError('release exists; reconcile it, never prepare replacement')
    protocol=json.loads(PROFILE.read_text());validate_hybrid_protocol(protocol)
    panel=json.loads((PARENT/'source_panel/panel_cases.json').read_text())
    historical=json.loads((PARENT/'outputs_sha256.json').read_text())
    assert ops.sha(PARENT/'source_panel/panel_cases.json')==historical['source_panel/panel_cases.json']
    staging=session/'bundle';inputs=staging/'inputs';inputs.mkdir(parents=True)
    selected=[]
    for task in TASKS:
        # Source membership precedes all dataset hashing and reads.
        spec=protocol['tasks'][task]
        assert spec['role']=='source'
        case=next(c for c in panel['cases'] if c['task']==task and c['coverage']=='early')
        assert case['demo'] in spec['source']['source_validation']
        path=ops.ROOT/'data/assets/libero/libero_goal'/f'{task}_demo.hdf5'
        assert ops.sha(path)==spec['source_sha256'],'source dataset changed'
        with h5py.File(path,'r') as handle:
            obs=handle[f"data/demo_{case['demo']}/obs"]
            assert 'agentview_rgb' in obs
            recorded=np.asarray(obs['agentview_rgb'][case['t']:case['t']+101],dtype=np.uint8)
        observation=recorded[0].copy()
        rel=f"source_panel/clips/{case['case_id'].replace(':','_')}/case.json"
        assert ops.sha(PARENT/rel)==historical[rel]
        old=json.loads((PARENT/rel).read_text())
        metadata_rel=f"generation_cache/{old['cache_key'][:2]}/{old['cache_key']}/metadata.json"
        assert ops.sha(PARENT/metadata_rel)==historical[metadata_rel]
        meta=json.loads((PARENT/metadata_rel).read_text())
        assert hashlib.sha256(resize_rgb(to_display(observation),(256,256)).tobytes()).hexdigest()==meta['conditioning_sha256'],'original conditioning image differs'
        name=case['case_id'].replace(':','_')+'.npz'
        np.savez_compressed(inputs/name,observation=observation,recorded=recorded)
        selected.append({k:v for k,v in case.items() if k!='path'}|{
            'input':name,'input_sha256':ops.sha(inputs/name),'observation_sha256':hashlib.sha256(observation.tobytes()).hexdigest(),
            'dataset_sha256':spec['source_sha256'],'historical_case_sha256':historical[rel],
            'historical_conditioning_sha256':meta['conditioning_sha256'],
            'historical_generation_seconds':old['generation_seconds']})
    ops.atomic(inputs/'cases.json',{'cases':selected,'protocol_digest':protocol['digest'],
        'parent_panel_sha256':historical['source_panel/panel_cases.json'],'scientific_admission':False})
    for name in MODULES:
        target=staging/'repo/raid_experiments'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ops.ROOT/'raid_experiments'/name,target)
    for name in ('requirements-wm-cosmos.lock','requirements-wm-sim.lock'):
        shutil.copyfile(ops.ROOT/name,staging/'repo'/name)
    shutil.copyfile(PROFILE,staging/'repo/protocol.json')
    rows=[{'path':p.relative_to(staging).as_posix(),'sha256':ops.sha(p),'size_bytes':p.stat().st_size}
          for p in sorted(staging.rglob('*')) if p.is_file()]
    manifest={'files':rows,'stage':'four_case_source_diagnostic','file_count':len(rows),'unpacked_bytes':sum(r['size_bytes'] for r in rows)}
    ops.atomic(staging/'package_manifest.json',manifest)
    archive=session/'source_panel_package.tar'
    with tarfile.open(archive,'w') as tar:
        for row in rows:tar.add(staging/row['path'],arcname=row['path'],recursive=False)
        tar.add(staging/'package_manifest.json',arcname='package_manifest.json',recursive=False)
    manifest.update(archive_sha256=ops.sha(archive),archive_bytes=archive.stat().st_size)
    ops.atomic(session/'package_release.json',manifest)
    ident='v6-source-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    root='/workspace/results/'+ident
    code={r['path']:r['sha256'] for r in rows if r['path'].endswith('.py')}
    for name in managed.HOST_FILES+('raid_experiments/cosmos_diagnostic_session.py',):code['host/'+name]=ops.sha(ops.ROOT/name)
    bindings={'code':hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest(),
        'protocol':ops.sha(PROFILE),'runtime':hashlib.sha256((ops.IMAGE_REFERENCE+ops.sha(ops.ROOT/'requirements-wm-cosmos.lock')+ops.sha(ops.ROOT/'requirements-wm-sim.lock')).encode()).hexdigest()}
    from .wm_pod import bootstrap_script,runtime_exports,runtime_python
    prefix=runtime_exports(root)+f'export HF_HOME=/root/raid-model-cache\nexport RAID_CONTAINER_IMAGE_DIGEST={ops.IMAGE_REFERENCE.split("@")[1]}\ncd {shlex.quote(root+"/repo")}\n'
    stages=[{'name':'archive_verify','command':managed.archive_verification_command(root,manifest['archive_sha256']),'resume_safe':True},
        {'name':'package_preflight','command':preflight(root,manifest['archive_sha256'])},
        {'name':'bootstrap','command':bootstrap_script(root),'resume_safe':True},
        {'name':'four_case_generation','command':prefix+f'{runtime_python(root)} -m raid_experiments.cosmos_diagnostic --protocol {root}/repo/protocol.json --cases {root}/inputs/cases.json --output {root}/collection/diagnostic'}]
    identity={'session_id':ident,'stage':'four_case_source_diagnostic','package_sha256':manifest['archive_sha256'],
        'bindings':bindings,'protocol_digest':protocol['digest'],'model_digest':protocol['model_digest'],
        'case_ids':[c['case_id'] for c in selected],
        'stage_commands':{s['name']:hashlib.sha256(s['command'].encode()).hexdigest() for s in stages},
        'scientific_result':False,'held_out_claims_eligible':False}
    ops.atomic(session/'release_identity.json',identity);ops.atomic(session/'code_binding_files.json',code)
    release={'session_id':ident,'root':root,'archive':str(archive.resolve()),'archive_sha256':manifest['archive_sha256'],
        'bindings':bindings,'identity_sha256':ops.sha(session/'release_identity.json'),'stages':stages}
    ops.atomic(session/'release.json',release);return release


def launch(session,acceptance_id,*,deposits='75',cap='5.50',volume_gb=80,min_gpu_memory_gb=48):
    session=Path(session)
    if (session/'intent.json').exists():raise RuntimeError('existing intent; resume without replacement')
    release=json.loads((session/'release.json').read_text())
    from .huggingface_auth import verify_access
    ops.atomic(session/'huggingface_access.json',verify_access())
    # Bound the whole setup/test/collection request, including storage and failures.
    admission=ops.quote(session,observed_deposits_usd=deposits,cap_usd=cap,compute_seconds=7200,storage_seconds=7800,volume_gb=volume_gb,min_gpu_memory_gb=min_gpu_memory_gb)
    ops.atomic(session/'s3_preflight.json',ops.verify_recovery_access(admission['data_center_id']))
    health=ops.service('/v2/health');now=int(time.time()*1000)
    body={'sessionId':release['session_id'],'stage':'source_panel','quoteStartAt':now,
        'collectBy':now+6300000,'deadlineAt':now+7200000,'deleteStorageAt':now+7800000,
        'capUsd':cap,'maxHourlyUsd':admission['gpu_hourly_usd'],'packageSha256':release['archive_sha256'],
        'bindings':{**release['bindings'],'deployment':health['deployment']},'acceptanceSessionId':acceptance_id,
        'sentinel':{'key':f"results/{release['session_id']}/collection/outputs_sha256.json",'sha256':release['identity_sha256']},
        'volumeSpec':{'size':volume_gb,'dataCenterId':admission['data_center_id']},
        'podSpec':{'imageName':ops.IMAGE_REFERENCE,'gpuTypeIds':[admission['gpu']],'gpuCount':1,
            'cloudType':'SECURE','computeType':'GPU','dataCenterIds':[admission['data_center_id']],
            'containerDiskInGb':60,'volumeInGb':0,'volumeMountPath':'/workspace','ports':['22/tcp'],
            'supportPublicIp':True,'env':{'PYTHONHASHSEED':'42','CUBLAS_WORKSPACE_CONFIG':':4096:8',
            'RAID_CONTAINER_IMAGE_DIGEST':ops.IMAGE_REFERENCE.split('@')[1]}},'spendAdmission':admission}
    ops.atomic(session/'intent.json',body)
    registered=ops.service('/v2/sessions/'+body['sessionId']+'/register',body);ops.atomic(session/'registered.json',registered)
    assert all(registered['request'][key]==body[key] for key in ('collectBy','deadlineAt','deleteStorageAt'))
    assert registered['alarmAt']==body['collectBy']
    ops.atomic(session/'launch_intent.json',{'fingerprint':registered['fingerprint'],'at':time.time()})
    result=ops.service('/v2/sessions/'+body['sessionId']+'/launch',{'fingerprint':registered['fingerprint']});ops.atomic(session/'launch_response.json',result)
    return managed.resume(session,setup=True)
