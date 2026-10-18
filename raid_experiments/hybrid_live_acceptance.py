"""Verify coordinator authority over live acceptance, without pod credentials.

Local JSON is only a locator. The trusted HTTPS service must independently
return the completed acceptance for the exact files executing this release.
"""
from __future__ import annotations
import hashlib,json,re,ssl,time,urllib.request
from pathlib import Path,PurePosixPath

AUTHORITY='https://raid-runpod-cutoff.constantinertel.workers.dev'
IMAGE='runpod/pytorch@sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f'
REQUIRED_CODE={
 'repo/raid_experiments/'+name+'.py' for name in (
  'hybrid_live_acceptance','hybrid_admission','hybrid_cli','hybrid_runner',
  'hybrid_protocol','hybrid_rollout','hybrid_study','hybrid_collect','hybrid_panel',
  'hybrid_heads','hybrid_retrieval','hybrid_analysis','world_model','models',
  'retrieval','wm_protocol','wm_rollout','source_worker')}
REQUIRED_CODE.add('repo/src/mae_encoder.py')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()

def read(path):
 value=json.loads(Path(path).read_text())
 if not isinstance(value,dict):raise ValueError('acceptance release record must be an object')
 return value

def fetch_evidence(session_id):
 if not isinstance(session_id,str) or not re.fullmatch(r'[a-zA-Z0-9_-]{8,80}',session_id):
  raise ValueError('invalid live acceptance session identity')
 url=AUTHORITY+'/v2/acceptance-evidence/'+session_id
 context=ssl.create_default_context()
 try:
  import certifi
  context=ssl.create_default_context(cafile=certifi.where())
 except ImportError:pass
 request=urllib.request.Request(url,headers={'User-Agent':'raid-v6-scientific-admission/1','Accept':'application/json'})
 with urllib.request.urlopen(request,context=context,timeout=10) as response:
  if response.geturl()!=url:raise ValueError('acceptance authority redirected')
  raw=response.read(16385)
 if len(raw)>16384:raise ValueError('acceptance evidence exceeds size limit')
 value=json.loads(raw)
 if not isinstance(value,dict):raise ValueError('invalid authoritative acceptance evidence')
 return value

def verify_live_acceptance(protocol,evidence_root,operational,*,repo_root=None):
 root=Path(evidence_root).resolve();repo=Path(repo_root or Path(__file__).resolve().parents[1]).resolve()
 identity_path=(root/'launch_identity.json').resolve();binding_path=(root/'launch_code_binding_files.json').resolve()
 if not identity_path.is_relative_to(root) or not binding_path.is_relative_to(root):
  raise ValueError('acceptance release metadata escapes collection root')
 identity=read(identity_path);files=read(binding_path)
 if not REQUIRED_CODE.issubset(files):raise ValueError('acceptance does not cover executing scientific code')
 for name,expected in files.items():
  parts=PurePosixPath(name)
  if parts.is_absolute() or '..' in parts.parts or '\\' in name or parts.suffix!='.py' or not re.fullmatch('[a-f0-9]{64}',str(expected)):
   raise ValueError('unsafe acceptance code binding')
  if name.startswith('repo/'):
   path=(repo/name.removeprefix('repo/')).resolve()
   if not path.is_relative_to(repo) or sha(path)!=expected:raise ValueError('executing code differs from accepted release: '+name)
  elif name.startswith('inputs/libero_repo/'):
   # The code binding also covers pinned LIBERO Python source. Pod staging
   # places it beside repo; local collection verification uses the pinned
   # checkout. Neither branch opens any task dataset or final-task file.
   staged=repo.parent/'inputs/libero_repo'
   library=(staged if staged.is_dir() else repo/'data/assets/libero/upstream').resolve()
   path=(library/name.removeprefix('inputs/libero_repo/')).resolve()
   if not path.is_relative_to(library) or sha(path)!=expected:
    raise ValueError('executing LIBERO code differs from accepted release: '+name)
  elif not name.startswith('host/'):raise ValueError('unknown acceptance code binding namespace')
 code=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
 original_path=repo/'artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json'
 original=read(original_path)
 if original.get('digest')!=protocol.get('digest') or identity.get('protocol_digest')!=protocol.get('digest') or identity.get('model_digest')!=protocol.get('model_digest'):
  raise ValueError('intended launch scientific identity mismatch')
 runtime=hashlib.sha256((IMAGE+sha(repo/'requirements-wm-cosmos.lock')+sha(repo/'requirements-wm-sim.lock')).encode()).hexdigest()
 expected={'packageSha256':identity['package_sha256'],'bindings':{'code':code,'protocol':sha(original_path),'runtime':runtime},'identitySha256':sha(identity_path)}
 if identity.get('bindings')!=expected['bindings']:raise ValueError('intended launch bindings differ from actual release files')
 if (operational.get('source_code_sha256')!=code
     or operational.get('package_sha256')!=expected['packageSha256']
     or operational.get('runtime_sha256')!=runtime):
  raise ValueError('operational record differs from intended release bindings')
 capsule=fetch_evidence(operational.get('acceptance_session_id'))
 if capsule.get('type')!='raid-v6-live-acceptance-v1' or capsule.get('sessionId')!=operational.get('acceptance_session_id') or capsule.get('acceptedRelease')!=expected or capsule.get('imageReference')!=IMAGE:
  raise ValueError('authoritative acceptance differs from intended launch')
 if hashlib.sha256(str(capsule.get('deployment','')).encode()).hexdigest()!=operational.get('service_deployment_sha256'):
  raise ValueError('acceptance deployment mismatch')
 if capsule.get('recoveredSha256')!=capsule.get('collectionSha256') or capsule.get('recoveredSha256')!=operational.get('expected_artifact_sha256') or capsule.get('terminationReason')!='independent_alarm':
  raise ValueError('authoritative independent recovery correspondence failed')
 try:age=time.time()-int(capsule['completedAt'])/1000
 except (KeyError,TypeError,ValueError):raise ValueError('acceptance completion time missing') from None
 if not 0<=age<=86400:raise ValueError('authoritative acceptance expired')
 return {'authority':AUTHORITY,'session_id':capsule['sessionId'],'deployment':capsule['deployment'],'code_sha256':code,'scientific_result':False}
