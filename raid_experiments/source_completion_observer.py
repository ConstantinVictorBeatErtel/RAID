"""Release completed source compute before a potentially long S3 collection.

This Mac observer is supplementary. It never creates resources, changes an
allocation, or issues a collection receipt. Independent service alarms remain
authoritative if this process disappears.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
from . import managed_source as source, operational_acceptance as ops


def read_state(session):
    # Avoid sharing the primary collector's status.json temporary filename.
    intent=json.loads((Path(session)/'intent.json').read_text())
    return ops.service('/v2/sessions/'+intent['sessionId'])


def observe(session):
    session=Path(session)
    state=read_state(session)
    if state.get('closed'):return {'state':'closed'}
    if state.get('terminating'):return {'state':'terminating'}
    if not state.get('volumeId') or not state.get('podId'):return {'state':'reconciling_creation'}
    release=json.loads((session/'release.json').read_text())
    request=state['request']
    if (request['stage']!='source_panel' or request['sessionId']!=release['session_id']
            or request['packageSha256']!=release['archive_sha256']
            or any(request['bindings'].get(key)!=value for key,value in release['bindings'].items())):
        raise ValueError('source observer resource/release ownership mismatch')
    client=source.s3_client(state)
    prefix=request['sentinel']['key'].removesuffix('outputs_sha256.json')
    def read(name,limit):
        body=client.get_object(Bucket=state['volumeId'],Key=prefix+name)['Body']
        try:raw=body.read(limit+1)
        finally:body.close()
        if len(raw)>limit:raise ValueError('source observer record oversized')
        return raw
    for marker in ('DONE','FAILED'):
        try:raw=read(marker,4096)
        except client.exceptions.NoSuchKey:continue
        record=json.loads(raw)
        manifest_raw=read('outputs_sha256.json',1024*1024)
        manifest=json.loads(manifest_raw)
        digest=hashlib.sha256(manifest_raw).hexdigest()
        if (record.get('manifest_sha256')!=digest
                or record.get('identity_sha256')!=release['identity_sha256']
                or manifest.get('release_identity.json')!=release['identity_sha256']
                or record.get('scientific_result') is not False):
            raise ValueError('source observer completion/release mismatch')
        stages={stage['name'] for stage in release['stages']}
        if marker=='DONE':
            if record.get('stages')!=len(stages) or any(
                    f'logs/{name}.exit.json' not in manifest or f'logs/{name}.log' not in manifest
                    for name in stages):
                raise ValueError('source observer stage completion incomplete')
        elif record.get('stage') not in stages or not record.get('exit_code'):
            raise ValueError('source observer failure identity invalid')
        intent={'marker':marker,'manifest_sha256':digest,'session_id':release['session_id'],
                'pod_id':state['podId'],'at':time.time(),'scientific_result':False,
                'reason':'closed committed worker result; release compute before downloading artifacts'}
        ops.atomic(session/'early_compute_release_intent.json',intent)
        ops.service('/v2/sessions/'+release['session_id']+'/terminate',{})
        return {'state':'termination_requested',**intent}
    return {'state':'worker_not_yet_committed'}


def watch(session,seconds=14400):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        try:
            result=observe(session)
            ops.atomic(Path(session)/'completion_observer_status.json',result)
            if result['state'] in {'closed','terminating','termination_requested'}:return result
        except Exception as error:
            ops.atomic(Path(session)/'completion_observer_error.json',{'error':str(error),'at':time.time()})
        time.sleep(15)
    return {'state':'observer_timeout','independent_cleanup_remains_active':True}


if __name__=='__main__':
    import sys
    print(json.dumps(watch(Path(sys.argv[1]))),flush=True)
