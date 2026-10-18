"""Real, bounded operational acceptance. Scientific admission remains separate.

No local or pod process can terminate this test automatically. The external
Durable Object owns cutoff, S3 recovery, and funded storage expiry. Resume only
observes the original allocation and never creates replacement compute.
"""
from __future__ import annotations
import base64
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import os
from pathlib import Path
import ssl
import subprocess
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path.home()/'.config/raid-hybrid-v6'
# Same pinned image as wm_validate; no scientific imports in the operational CLI.
IMAGE_REFERENCE = 'runpod/pytorch@sha256:60baa36d3fb6b98fd4f4ece6b96776c83c01a8b7c540e54460ab4d496816141f'
REGIONS = {'EU-CZ-1','EU-RO-1','EUR-IS-1','EUR-NO-1','US-CA-2','US-GA-2','US-IL-1','US-KS-2','US-MD-1','US-MO-1','US-MO-2','US-NC-1','US-NC-2','US-NE-1','US-WA-1'}
FILES = ['raid_experiments/operational_acceptance.py','scripts/pod_wm_session.py',
         'services/runpod-cutoff/src/index.ts','services/runpod-cutoff/src/managed.ts',
         'services/runpod-cutoff/package-lock.json','services/runpod-cutoff/wrangler.jsonc',
         'requirements-wm-cosmos.lock','requirements-wm-sim.lock',
         'artifacts/raid_hybrid_libero_goal_v6/protocol_v6.json']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def cli(*args):
    p=subprocess.run(['runpodctl',*args],check=True,capture_output=True,text=True,timeout=45)
    return json.loads(p.stdout,parse_float=str)

def storage_catalog():
    """Read live storage support; a documented S3 endpoint is not placement stock."""
    import certifi
    import tomllib
    token=tomllib.loads((Path.home()/'.runpod/config.toml').read_text())['apikey']
    req=urllib.request.Request('https://api.runpod.io/v2/catalog/datacenters?include=GPU_AVAILABILITY',
        headers={'Authorization':'Bearer '+token,'User-Agent':'raid-v6-provider-placement/1'})
    with urllib.request.urlopen(req,context=ssl.create_default_context(cafile=certifi.where()),timeout=15) as response:
        result=json.load(response)
    rows=result.get('dataCenters')
    if not isinstance(rows,list) or not rows:raise RuntimeError('live storage catalog unavailable')
    return result

def compatible_placements(rows,catalog):
    """Require priced GPU stock, live standard storage and direct S3 recovery."""
    centers={}
    for center in catalog.get('dataCenters',[]):
        ident=center.get('id')
        if not ident or ident in centers:raise RuntimeError('invalid or duplicate data center catalog')
        centers[ident]=center
    if not centers:raise RuntimeError('live storage catalog unavailable')
    candidates=[]
    families=('A100','A40','A30','L40','H100','H200','RTX 6000 Ada','RTX A6000','B200')
    for row in rows:
        name=row.get('gpuId','')
        if not row.get('available') or not row.get('secureCloud') or int(row.get('memoryInGb',0))<48 or not any(f.lower() in name.lower() for f in families):continue
        try:rate=Decimal(str(row.get('securePricePerHr')))
        except Exception:raise RuntimeError('compatible GPU price missing or invalid') from None
        if not rate.is_finite() or rate<=0:raise RuntimeError('compatible GPU price missing or invalid')
        for region in row.get('dataCenterAvailability',[]):
            ident=region.get('dataCenterId');center=centers.get(ident,{})
            live_gpu=any(g.get('id')==name and str(g.get('availability','')).lower() in {'low','medium','high'} for g in center.get('gpuAvailability',[]))
            # v1 creation uses the region's default tier; mixed tiers do not
            # establish that the registered STANDARD price applies.
            if ident in REGIONS and center.get('networkVolumeTypes')==['STANDARD'] and live_gpu and str(region.get('stockStatus','none')).lower() in {'low','medium','high'}:
                candidates.append((rate,ident,name))
    return sorted(set(candidates))

def service(path, body=None):
    import certifi
    url=(CONFIG/'cutoff-url').read_text().strip().rstrip('/')
    req=urllib.request.Request(url+path,headers={'Authorization':'Bearer '+(CONFIG/'cutoff-client-token').read_text().strip(),'Content-Type':'application/json','User-Agent':'raid-v6-operational-acceptance/1'},data=None if body is None else json.dumps(body).encode())
    context=ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(req,context=context,timeout=25) as r:return json.load(r)
    except __import__('urllib.error',fromlist=['HTTPError']).HTTPError as exc:
        # Service emits structured errors, never secret values.
        raise RuntimeError(f'cutoff HTTP {exc.code}: {exc.read().decode()[:500]}') from None

# Billing may list slightly MORE charges than the balance drop (e.g. rounded storage hours). Spend accounting
# uses the larger billed total, so this direction is conservative. A driver may widen it with a recorded reason.
OVERSTATED_BILLING_TOLERANCE_USD=Decimal('0.01')

def ledger_reserved(allocation):
    """Same rule as the service ledger: a closed session counts at the cost the service recorded when it cleaned
    it; open sessions, and closed sessions without a recorded cost, count at their cap."""
    return sum((Decimal(r['spentUsd']) if r.get('clean') and r.get('spentUsd') else Decimal(r['cap'])
                for r in allocation.get('reservations',{}).values()),Decimal(0))

def quote(session, *, observed_deposits_usd='45',cap_usd='2',compute_seconds=900,storage_seconds=1200,volume_gb=20,min_gpu_memory_gb=48):
    """Reconcile deposits, current credit and lifetime charges; reserve rounding risk."""
    from .spending import reconcile_runpod_spending
    # Existing user disclosure: no additional RAID spending outside this account.
    # Provider console: four reloads (10+10+15+10), autopay disabled; free CF plan.
    record=reconcile_runpod_spending('0','0')
    account=cli('user');pods=cli('pod','list','--all');volumes=cli('network-volume','list')
    if pods or volumes or Decimal(str(account['currentSpendPerHr'])) != 0:
        raise RuntimeError('pre-existing resources or spend require reconciliation')
    balance=Decimal(str(account['clientBalance']))
    allocation=service('/v2/allocation');uncertain=Decimal(0);known=Decimal(0)
    permitted=allocation.get('retryAuthorization',{}).get('deferredSessions',[])
    for ident,reservation in allocation.get('reservations',{}).items():
        # A clean reservation with a service-recorded cost was cleaned only after verified deletion; skip its read.
        if reservation.get('clean') and reservation.get('spentUsd'):known+=Decimal(reservation['cap']);continue
        history=service('/v2/sessions/'+ident)
        if not reservation.get('clean'):
            if ident not in permitted or history.get('podRequested') or history.get('podId') or history.get('volumeId') or not history.get('podDeleted') or not history.get('terminating'):
                raise RuntimeError('unresolved prior allocation without bounded retry authorization')
            uncertain+=Decimal(reservation['cap'])
        elif not history.get('closed') or not history.get('podDeleted') or not history.get('volumeDeleted'):
            raise RuntimeError('outstanding liability cannot be attributed to cleaned resources')
        known+=Decimal(reservation['cap'])
    discrepancy=Decimal(observed_deposits_usd)-balance-Decimal(record['provider_charges_usd'])
    if discrepancy < -OVERSTATED_BILLING_TOLERANCE_USD:
        raise RuntimeError(f'deposits/credit/billing do not reconcile: USD {discrepancy}')
    if discrepancy < 0:record['billing_overstates_balance_drop_usd']=str(-discrepancy)
    if discrepancy>max(Decimal('0.01'),known):raise RuntimeError('credit/billing discrepancy exceeds known bounded prior allocations')
    # Explicit reserve covers both the positive billing residual and precision noise.
    liability=uncertain+max(Decimal('0.01'),discrepancy)
    record['outstanding_allowance_usd']=str(liability)
    record['outstanding_evidence']={'deposit_credits_usd':observed_deposits_usd,'balance_usd':str(balance),'billing_discrepancy_usd':str(discrepancy),'reserve_usd':str(liability),'basis':'console deposits; no resources/spend since retained earlier snapshots, billing delay one hour; conservative one-cent residual reserve'}
    record['outstanding_evidence']['uncertain_creation_reserve_usd']=str(uncertain)
    record['outstanding_evidence']['retry_authorization']=allocation.get('retryAuthorization')
    record['total_spent_usd']=str(Decimal(record['provider_charges_usd'])+liability)
    from .spending import _canonical_digest
    record['spending_digest']=_canonical_digest(record)
    atomic(session/'billing_before.json',record)
    catalog=storage_catalog();atomic(session/'placement_catalog.json',catalog)
    if not isinstance(min_gpu_memory_gb,int) or min_gpu_memory_gb<48:
        raise ValueError('invalid minimum GPU memory')
    rows=cli('gpu','list');candidates=compatible_placements(
        [r for r in rows if int(r.get('memoryInGb',0))>=min_gpu_memory_gb],catalog)
    if not candidates:raise RuntimeError('no available compatible GPU with live standard storage and direct S3 recovery')
    rate,region,gpu=min(candidates)
    now=datetime.now(timezone.utc).isoformat()
    # Worst-case compute through funded deadline, plus rounded-up hourly volume storage.
    hours=Decimal(str(compute_seconds))/3600
    storage_hours=(Decimal(str(storage_seconds))/3600).to_integral_value(rounding=ROUND_CEILING)
    cap=Decimal(cap_usd)
    projected=rate*hours+Decimal(60)*Decimal('0.10')*hours/720+Decimal(volume_gb)*Decimal('0.07')*storage_hours/720
    # Cloudflare Free plan and Runpod no transfer fees were verified from account/docs.
    # Retain a separate $0.10 service/rounding contingency despite no incremental charge.
    projected+=Decimal('0.10')
    remaining=Decimal('225')-Decimal(record['total_spent_usd'])
    reserved=ledger_reserved(allocation)
    if not cap.is_finite() or cap<=0 or cap+reserved>Decimal(allocation.get('cumulativeCapUsd','10')) or min(balance-liability,remaining)<cap or projected>cap:
        raise RuntimeError('stage reservation exceeds prepaid, project or shared source allocation')
    result={'status':'passed','currency':'USD','all_liabilities_known':True,'provider_inventory_checked':True,'refreshed_at':now,'billing_refreshed_at':record['collected_at'],'inventory_refreshed_at':now,'gpu':gpu,'gpu_hourly_usd':str(rate),'network_volume_gb_month_usd':'0.07','container_disk_gb_month_usd':'0.10','container_disk_gb':60,'transfer_reserve_usd':'0','cutoff_service_cost_reserve_usd':'0.10','prepaid_available_usd':str(balance),'remaining_project_allowance_usd':str(remaining),'total_study_spend_usd':record['total_spent_usd'],'project_ceiling_usd':'250','project_reserve_usd':'25','projected_source_cost_usd':format(projected,'f'),'data_center_id':region,'network_volume_size_gb':20,'source_cap_usd':'2','billing_digest':record['spending_digest'],'quote_provenance':{'gpu':'runpodctl gpu list','storage_and_transfer':'https://docs.runpod.io/pods/pricing','network_volume_billing_unit':'hour, conservatively rounded up','container_disk_billing_unit':'second','monthly_hour_denominator':720,'service':'Cloudflare authorized account Workers Free, verified in dashboard; $0.10 contingency reserved'}}
    result.update(network_volume_size_gb=volume_gb,source_cap_usd=cap_usd)
    result['prepaid_available_usd']=str(balance-liability)
    result['gross_prepaid_balance_usd']=str(balance)
    result['placement_evidence']={'catalog_url':'https://api.runpod.io/v2/catalog/datacenters?include=GPU_AVAILABILITY','refreshed_at':now,'sha256':sha(session/'placement_catalog.json'),'storage_tier':'STANDARD','live_gpu_stock_checked':True}
    atomic(session/'quote.json',result);return result

def plan():
    return {'mode':'operational-acceptance','image':IMAGE_REFERENCE,'compute_cutoff_seconds':480,'funded_compute_deadline_seconds':900,'storage_expiry_seconds':1200,'reservation_usd':'2','shared_source_allowance_usd':'10','shared_source_deadline_hours':2,'network_volume_gb':20,'pod_watchdogs':False,'local_shutdown_supervisor':False,'scientific_result':False}

def verify_recovery_access(region):
    import boto3
    from botocore.config import Config
    credentials=dict(tuple(part.strip() for part in line.split('=',1)) for line in (CONFIG/'runpod-s3-credentials.txt').read_text().splitlines() if '=' in line)
    client=boto3.client('s3',region_name=region,endpoint_url=f'https://s3api-{region.lower()}.runpod.io',
        aws_access_key_id=credentials['aws_access_key_id'].strip(),aws_secret_access_key=credentials['aws_secret_access_key'].strip(),
        config=Config(connect_timeout=10,read_timeout=10,retries={'max_attempts':0}))
    result=client.list_buckets()
    if result['ResponseMetadata']['HTTPStatusCode']!=200:raise RuntimeError('S3 recovery authentication failed')
    return {'region':region,'authenticated':True,'at':datetime.now(timezone.utc).isoformat()}

def launch(session:Path,*,scientific_release=None,timing=None,cap_usd='2',observed_deposits_usd=None):
    session=Path(session)
    if (session/'intent.json').exists():raise RuntimeError('session already has launch intent; use status/resume, never launch again')
    session.mkdir(parents=True,exist_ok=True)
    health=service('/v2/health')
    timing=timing or {'cutoff':480,'compute':900,'storage':1200}
    quote_options={} if observed_deposits_usd is None else {'observed_deposits_usd':observed_deposits_usd}
    admission=quote(session) if not quote_options and timing=={'cutoff':480,'compute':900,'storage':1200} and cap_usd=='2' else quote(session,cap_usd=cap_usd,compute_seconds=timing['compute'],storage_seconds=timing['storage'],**quote_options)
    atomic(session/'s3_preflight.json',verify_recovery_access(admission['data_center_id']))
    files={rel:{'sha256':sha(ROOT/rel),'size':(ROOT/rel).stat().st_size} for rel in FILES}
    package=json.dumps(files,sort_keys=True,separators=(',',':')).encode();package_hash=hashlib.sha256(package).hexdigest()
    atomic(session/'package_manifest.json',{'files':files,'sha256':package_hash,'stage':'operational_acceptance','dataset_access':'none'})
    ident='v6-accept-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:8]
    content=(json.dumps({'session_id':ident,'package_sha256':package_hash,'kind':'real_runpod_operational_sentinel','scientific_result':False},sort_keys=True)+'\n').encode()
    key=f'acceptance/{ident}/sentinel.json';sentinel_hash=hashlib.sha256(content).hexdigest()
    encoded=base64.b64encode(content).decode()
    # Atomic publication is performed by the actual container on its attached volume.
    script=f'''set -eu
python3 - <<'SENTINEL'
import base64, pathlib, os
p=pathlib.Path('/workspace/{key}')
p.parent.mkdir(parents=True,exist_ok=True)
f=p.with_suffix('.tmp')
with f.open('wb') as h:
 h.write(base64.b64decode('{encoded}'));h.flush();os.fsync(h.fileno())
f.replace(p)
d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d)
print('RAID_ACCEPTANCE_SENTINEL_COMMITTED',flush=True)
SENTINEL
exec sleep 7200
'''
    now=int(time.time()*1000)
    b={'sessionId':ident,'stage':'operational_acceptance','quoteStartAt':now,'collectBy':now+480000,'deadlineAt':now+900000,'deleteStorageAt':now+1200000,'capUsd':'2','maxHourlyUsd':admission['gpu_hourly_usd'],'packageSha256':package_hash,'bindings':{'code':hashlib.sha256(json.dumps({r:v for r,v in files.items() if r.endswith(('.py','.ts'))},sort_keys=True).encode()).hexdigest(),'protocol':files[FILES[-1]]['sha256'],'runtime':hashlib.sha256((IMAGE_REFERENCE+files['requirements-wm-cosmos.lock']['sha256']+files['requirements-wm-sim.lock']['sha256']).encode()).hexdigest(),'deployment':health['deployment']},'sentinel':{'key':key,'sha256':sentinel_hash},'volumeSpec':{'size':20,'dataCenterId':admission['data_center_id']},'podSpec':{'imageName':IMAGE_REFERENCE,'gpuTypeIds':[admission['gpu']],'gpuCount':1,'cloudType':'SECURE','computeType':'GPU','dataCenterIds':[admission['data_center_id']],'containerDiskInGb':60,'volumeInGb':0,'volumeMountPath':'/workspace','dockerEntrypoint':['/bin/bash','-lc'],'dockerStartCmd':[script],'env':{}},'spendAdmission':admission}
    b.update(collectBy=now+timing['cutoff']*1000,deadlineAt=now+timing['compute']*1000,deleteStorageAt=now+timing['storage']*1000,capUsd=cap_usd)
    if scientific_release is not None:b['scientificRelease']=scientific_release
    atomic(session/'intent.json',b)
    registered=service('/v2/sessions/'+ident+'/register',b);atomic(session/'registered.json',registered)
    for key in ['collectBy','deadlineAt','deleteStorageAt']:
        if registered['request'][key]!=b[key]:raise RuntimeError('persisted deadline read-back mismatch')
    if registered['alarmAt']!=b['collectBy']:raise RuntimeError('independent cutoff alarm not armed')
    atomic(session/'launch_intent.json',{'session_id':ident,'fingerprint':registered['fingerprint'],'at':datetime.now(timezone.utc).isoformat()})
    # A timeout is deliberately not retried. Subsequent calls use status/resume only.
    result=service('/v2/sessions/'+ident+'/launch',{'fingerprint':registered['fingerprint']})
    atomic(session/'launch_response.json',result)
    return result

def status(session):
    b=json.loads((Path(session)/'intent.json').read_text());s=service('/v2/sessions/'+b['sessionId']);atomic(Path(session)/'status.json',s);return s

def collect(session):
    session=Path(session);s=status(session)
    if not s.get('podDeleted') or not s.get('recovered'):raise RuntimeError('independent termination/recovery not yet verified')
    body=s['recovered']['body'].encode();expected=s['request']['sentinel']['sha256']
    if hashlib.sha256(body).hexdigest()!=expected:raise RuntimeError('independent recovery checksum mismatch')
    atomic(session/'collected_sentinel.json',json.loads(body))
    # Keep exact original bytes for the receipt, not reserialized JSON.
    tmp=session/'sentinel.exact.tmp'
    with tmp.open('wb') as f:f.write(body);f.flush();os.fsync(f.fileno())
    tmp.replace(session/'sentinel.exact.json')
    service('/v2/sessions/'+s['request']['sessionId']+'/collection',{'sha256':expected})
    return status(session)

def dispatch(command,session,*,authorize=False,interval=10,max_wait=1500):
    if command=='plan':return plan()
    if command=='launch':
        if not authorize:raise RuntimeError('--authorize-spend required')
        return launch(Path(session))
    if command in ('status','resume'):return status(session)
    if command=='collect':return collect(session)
    if command=='teardown':
        s=status(session);service('/v2/sessions/'+s['request']['sessionId']+'/terminate',{});return status(session)
    if command=='watch':
        end=time.monotonic()+max_wait
        while time.monotonic()<end:
            s=status(session)
            if s.get('recovered') and not s.get('receipt'):s=collect(session)
            if s.get('closed'):return s
            time.sleep(max(1,min(interval,30)))
        raise RuntimeError('watch ended; independent service cleanup remains active')
    raise ValueError('unsupported operational command')
