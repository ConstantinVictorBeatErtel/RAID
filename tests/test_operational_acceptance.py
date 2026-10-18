import hashlib
import json
from pathlib import Path
import pytest
from raid_experiments import operational_acceptance as ops


def test_resume_only_observes_original_session(tmp_path, monkeypatch):
    ops.atomic(tmp_path/'intent.json',{'sessionId':'original-session'})
    calls=[]
    def service(path,body=None):
        calls.append((path,body));return {'request':{'sessionId':'original-session'},'launchAt':123}
    monkeypatch.setattr(ops,'service',service)
    assert ops.dispatch('resume',tmp_path)['launchAt']==123
    assert calls==[('/v2/sessions/original-session',None)]
    with pytest.raises(RuntimeError,match='already has launch intent'):ops.launch(tmp_path)


def test_invalid_recovery_cannot_replace_verified_local_copy(tmp_path,monkeypatch):
    old=tmp_path/'sentinel.exact.json';old.write_text('verified older copy')
    monkeypatch.setattr(ops,'status',lambda _: {'podDeleted':True,'recovered':{'body':'corrupt'},'request':{'sentinel':{'sha256':'a'*64}}})
    monkeypatch.setattr(ops,'service',lambda *a:pytest.fail('receipt must not be submitted'))
    with pytest.raises(RuntimeError,match='checksum'):ops.collect(tmp_path)
    assert old.read_text()=='verified older copy'


def test_launch_requires_persisted_matching_alarm_before_paid_request(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(ops,'quote',lambda _: {'gpu_hourly_usd':'1','gpu':'gpu','data_center_id':'US-WA-1'})
    monkeypatch.setattr(ops,'verify_recovery_access',lambda _: {'authenticated':True})
    def service(path,body=None):
        calls.append(path)
        if path=='/v2/health':return {'deployment':'release-version'}
        if path.endswith('/register'):return {'request':body,'alarmAt':0,'fingerprint':'a'*64}
        pytest.fail('paid launch must not happen')
    monkeypatch.setattr(ops,'service',service)
    with pytest.raises(RuntimeError,match='not armed'):ops.launch(tmp_path)
    assert (tmp_path/'intent.json').is_file()
    assert not any(p.endswith('/launch') for p in calls)


def test_collection_receipt_requires_exact_recovered_bytes(tmp_path,monkeypatch):
    body='{"sentinel": true}\n';digest=hashlib.sha256(body.encode()).hexdigest()
    state={'podDeleted':True,'recovered':{'body':body},'request':{'sessionId':'acceptance-case','sentinel':{'sha256':digest}}}
    monkeypatch.setattr(ops,'status',lambda _:state)
    calls=[];monkeypatch.setattr(ops,'service',lambda path,body:calls.append((path,body)))
    ops.collect(tmp_path)
    assert (tmp_path/'sentinel.exact.json').read_bytes()==body.encode()
    assert calls==[('/v2/sessions/acceptance-case/collection',{'sha256':digest})]


def test_operational_plan_has_no_scientific_or_local_shutdown_work():
    from raid_experiments.wm_pod import IMAGE_REFERENCE
    assert ops.IMAGE_REFERENCE==IMAGE_REFERENCE
    plan=ops.plan()
    assert plan['scientific_result'] is False
    assert plan['pod_watchdogs'] is False
    assert plan['local_shutdown_supervisor'] is False
    assert plan['compute_cutoff_seconds']<plan['funded_compute_deadline_seconds']<plan['storage_expiry_seconds']<7200
    assert plan['shared_source_allowance_usd']=='10'


def test_recovery_preflight_parses_spaced_credential_assignments(tmp_path,monkeypatch):
    import sys
    from types import SimpleNamespace
    (tmp_path/'runpod-s3-credentials.txt').write_text('aws_access_key_id = fake-access\naws_secret_access_key = fake-secret\n')
    monkeypatch.setattr(ops,'CONFIG',tmp_path)
    def client(service,**kwargs):
        assert service=='s3'
        assert kwargs['aws_access_key_id']=='fake-access'
        assert kwargs['aws_secret_access_key']=='fake-secret'
        assert kwargs['region_name']=='US-IL-1'
        assert kwargs['endpoint_url']=='https://s3api-us-il-1.runpod.io'
        return SimpleNamespace(list_buckets=lambda:{'ResponseMetadata':{'HTTPStatusCode':200}})
    monkeypatch.setitem(sys.modules,'boto3',SimpleNamespace(client=client))
    monkeypatch.setitem(sys.modules,'botocore.config',SimpleNamespace(Config=lambda **kw:kw))
    assert ops.verify_recovery_access('US-IL-1')['authenticated'] is True


def placement_rows():
    return [{'gpuId':'NVIDIA RTX 6000 Ada Generation','available':True,'secureCloud':True,
             'memoryInGb':48,'securePricePerHr':'0.99','dataCenterAvailability':[
                 {'dataCenterId':region,'stockStatus':'High'} for region in ('US-WA-1','US-IL-1','EU-RO-1')]}]


def placement_catalog():
    return {'dataCenters':[{'id':region,'networkVolumeTypes':tiers,'gpuAvailability':[
        {'id':'NVIDIA RTX 6000 Ada Generation','availability':'HIGH'}]}
        for region,tiers in [('US-WA-1',[]),('US-IL-1',['STANDARD']),('EU-RO-1',['STANDARD'])]]}


def test_placement_rejects_s3_region_without_live_storage_and_breaks_price_ties():
    candidates=ops.compatible_placements(placement_rows(),placement_catalog())
    assert [x[1] for x in candidates]==['EU-RO-1','US-IL-1']


@pytest.mark.parametrize('tiers',[['HIGH_PERFORMANCE'],['HIGH_PERFORMANCE','STANDARD'],['STANDARD','HIGH_PERFORMANCE']])
def test_placement_excludes_unpriced_high_performance_and_missing_gpu_stock(tiers):
    catalog=placement_catalog()
    catalog['dataCenters'][1]['networkVolumeTypes']=tiers
    catalog['dataCenters'][2]['gpuAvailability']=[]
    assert ops.compatible_placements(placement_rows(),catalog)==[]


@pytest.mark.parametrize('price',[None,'NaN','-1','0','Infinity'])
def test_placement_rejects_missing_or_invalid_prices(price):
    rows=placement_rows();rows[0]['securePricePerHr']=price
    with pytest.raises(RuntimeError,match='price missing or invalid'):
        ops.compatible_placements(rows,placement_catalog())


def test_placement_rejects_missing_or_duplicate_catalog():
    with pytest.raises(RuntimeError,match='catalog unavailable'):
        ops.compatible_placements(placement_rows(),{})
    catalog=placement_catalog();catalog['dataCenters'].append(catalog['dataCenters'][0])
    with pytest.raises(RuntimeError,match='duplicate'):
        ops.compatible_placements(placement_rows(),catalog)


def mocked_retry_quote(monkeypatch,authorized=True):
    import datetime
    from raid_experiments import spending
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    monkeypatch.setattr(spending,'reconcile_runpod_spending',lambda *a:{'provider_charges_usd':'32','collected_at':stamp})
    def cli(*args):
        if args==('user',):return {'clientBalance':'13','currentSpendPerHr':0}
        if args==('gpu','list'):return placement_rows()
        return []
    monkeypatch.setattr(ops,'cli',cli);monkeypatch.setattr(ops,'storage_catalog',placement_catalog)
    def service(path):
        if path=='/v2/allocation':return {'reservations':{'historical-clean':{'cap':'2','clean':True},'uncertain-500':{'cap':'1','clean':False}},'retryAuthorization':{'deferredSessions':['uncertain-500'] if authorized else []}}
        if path.endswith('historical-clean'):return {'closed':True,'podDeleted':True,'volumeDeleted':True}
        return {'podDeleted':True,'terminating':True}
    monkeypatch.setattr(ops,'service',service)


def test_retry_quote_retains_uncertain_liability_and_reconciles_funded_credit(tmp_path,monkeypatch):
    mocked_retry_quote(monkeypatch)
    quote=ops.quote(tmp_path,cap_usd='1')
    # The uncertain request and billing-delay reserve both reduce funded credit.
    assert quote['prepaid_available_usd']=='11.99'
    assert quote['gross_prepaid_balance_usd']=='13'
    record=json.loads((tmp_path/'billing_before.json').read_text())
    assert record['outstanding_allowance_usd']=='1.01'
    assert record['outstanding_evidence']['uncertain_creation_reserve_usd']=='1'


def test_retry_cannot_omit_uncertain_liability_or_reset_cumulative_money(tmp_path,monkeypatch):
    mocked_retry_quote(monkeypatch,authorized=False)
    with pytest.raises(RuntimeError,match='without bounded retry'):ops.quote(tmp_path)
    mocked_retry_quote(monkeypatch)
    with pytest.raises(RuntimeError,match='reservation exceeds'):ops.quote(tmp_path,cap_usd='8')
