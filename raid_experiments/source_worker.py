"""Detached source worker; stdlib only until locked setup completes.

Every stage closes its outputs before publishing a checksum manifest. Failed or
interrupted work remains diagnostic evidence, never a completed source panel.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def atomic(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(body,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def publish(collection,committed):
    # Only immutable, closed command outputs enter a published manifest.
    files={'release_identity.json':sha(collection/'release_identity.json')}
    for name in committed:
        path=collection/name
        if path.is_file():files[name]=sha(path)
    for path in sorted(collection.rglob('*')):
        rel=path.relative_to(collection).as_posix()
        if path.is_file() and not rel.startswith('logs/') and path.name not in {'worker.log','outputs_sha256.json','DONE','FAILED'} and not path.name.endswith('.tmp'):
            files[rel]=sha(path)
    atomic(collection/'outputs_sha256.json',files)
    return sha(collection/'outputs_sha256.json')

def configure_file_limit():
    """LIBERO mesh loading retains descriptors across repeated environment resets."""
    import resource
    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    if sys.platform != 'linux':
        return {'before': list(before), 'after': list(before)}
    soft, hard = before
    target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return {'before': list(before), 'after': list(resource.getrlimit(resource.RLIMIT_NOFILE))}

def run(config_path):
    import fcntl
    config=json.loads(Path(config_path).read_text());root=Path(config['root'])
    root.mkdir(parents=True,exist_ok=True)
    with (root/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('existing worker or orphaned stage is still running') from None
        # The active stage inherits this lock. A dead supervisor therefore
        # cannot start a second copy while its original training still runs.
        return _run(config_path,lock.fileno())

def _run(config_path,lock_fd):
    config=json.loads(Path(config_path).read_text());root=Path(config['root']);collection=root/'collection'
    collection.mkdir(parents=True,exist_ok=True)
    file_limit = configure_file_limit()
    identity=collection/'release_identity.json'
    if sha(identity)!=config['identity_sha256']:raise RuntimeError('worker release identity mismatch')
    registered=json.loads(identity.read_text())
    commands={stage['name']:hashlib.sha256(stage['command'].encode()).hexdigest()
              for stage in config['stages']}
    if len(commands)!=len(config['stages']):raise RuntimeError('duplicate worker stage identity')
    if registered.get('stage_commands')!=commands and registered.get('synthetic') is not True:
        raise RuntimeError('worker commands differ from registered release')
    import re
    if any(not re.fullmatch('[a-z][a-z0-9_]{0,63}',name) for name in commands):
        raise RuntimeError('unsafe worker stage identity')
    if (collection/'DONE').exists():raise RuntimeError('completed allocation cannot be rerun')
    worker_identity=root/'worker_identity.json'
    if worker_identity.exists() and json.loads(worker_identity.read_text()).get('config_sha256')!=sha(config_path):
        raise RuntimeError('worker configuration is immutable during resume')
    atomic(worker_identity,{'pid':os.getpid(),'config_sha256':sha(config_path),'started_at':time.time(),
                           'nofile_limit': file_limit})
    (root/'runner.pid').write_text(str(os.getpid())+'\n')
    # Keep diagnostics from every prior interruption, including when another
    # restart dies before the next command completes. Publish the closed release
    # and setup diagnostics before starting a potentially long first command.
    committed=[path.relative_to(collection).as_posix()
               for path in sorted((collection/'logs').glob('*.interrupted.*.log'))
               if path.is_file() and not path.is_symlink()]
    publish(collection,committed)
    for stage in config['stages']:
        name=stage['name'];log=collection/'logs'/f'{name}.log';exit_path=collection/'logs'/f'{name}.exit.json'
        if exit_path.exists():
            saved=json.loads(exit_path.read_text())
            if saved['command_sha256']!=hashlib.sha256(stage['command'].encode()).hexdigest() or saved['exit_code']!=0 or sha(log)!=saved['log_sha256']:
                raise RuntimeError('interrupted stage cannot be silently reused or restarted')
            committed.extend([log.relative_to(collection).as_posix(),exit_path.relative_to(collection).as_posix()]);continue
        remaining=config['compute_cutoff_ms']/1000-time.time()-30
        if remaining<=0:raise RuntimeError('original allocation compute cutoff reached')
        if log.exists():
            if not stage.get('resume_safe',False):
                raise RuntimeError('interrupted command requires a verified resumable stage')
            import uuid
            prior=log.with_name(f'{name}.interrupted.{uuid.uuid4().hex[:8]}.log')
            log.replace(prior)
            committed.append(prior.relative_to(collection).as_posix())
        log.parent.mkdir(parents=True,exist_ok=True);started=time.time();code=-1
        with log.open('w') as output:
            process=subprocess.Popen(['bash','-c',stage['command']],stdout=output,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
            atomic(root/'active_stage.json',{'name':name,'pid':process.pid,'process_group':process.pid,
                'worker_pid':os.getpid(),'command_sha256':hashlib.sha256(stage['command'].encode()).hexdigest(),
                'started_at':started,'config_sha256':sha(config_path)})
            try:code=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=124
            output.flush();os.fsync(output.fileno())
        atomic(exit_path,{'exit_code':code,'started_at':started,'finished_at':time.time(),'seconds':time.time()-started,'command_sha256':hashlib.sha256(stage['command'].encode()).hexdigest(),'log_sha256':sha(log)})
        committed.extend([log.relative_to(collection).as_posix(),exit_path.relative_to(collection).as_posix()])
        if code:
            digest=publish(collection,committed)
            atomic(collection/'FAILED',{'stage':name,'exit_code':code,
                'manifest_sha256':digest,'identity_sha256':config['identity_sha256'],
                'scientific_result':False})
            return code
        publish(collection,committed)
    digest=publish(collection,committed)
    atomic(collection/'DONE',{'manifest_sha256':digest,'identity_sha256':config['identity_sha256'],'stages':len(config['stages']),'scientific_result':False})
    return 0

if __name__=='__main__':
    raise SystemExit(run(sys.argv[1]))
