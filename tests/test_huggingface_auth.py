import shlex
import subprocess
import sys
from types import SimpleNamespace
import pytest
from raid_experiments import huggingface_auth as auth


def test_secret_installed_via_stdin_with_private_permissions(tmp_path,monkeypatch):
    secret='hf_syntheticCredentialForTest'
    monkeypatch.setattr(auth,'saved_token',lambda:secret)
    class Transport:
        def _ssh_argv(self,pod,command,*,stdin):
            assert pod=='owned' and stdin and secret not in command
            script=shlex.split(command)[2].replace('/root/raid-model-cache',str(tmp_path/'private'))
            return [sys.executable,'-c',script]
    auth.install('owned',Transport())
    token=tmp_path/'private/token'
    assert token.read_text()==secret
    assert token.stat().st_mode & 0o777 == 0o600
    assert token.parent.stat().st_mode & 0o777 == 0o700
    assert list(token.parent.iterdir())==[token]


def test_delivery_error_never_repeats_secret_output(monkeypatch):
    secret='hf_syntheticCredentialForTest'
    monkeypatch.setattr(auth,'saved_token',lambda:secret)
    monkeypatch.setattr(auth.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,stdout=secret,stderr=secret))
    transport=SimpleNamespace(_ssh_argv=lambda *a,**k:['ssh'])
    with pytest.raises(RuntimeError) as e:auth.install('owned',transport)
    assert secret not in str(e.value)


def test_access_denial_never_downloads_weights(monkeypatch):
    monkeypatch.setattr(auth,'saved_token',lambda:'hf_syntheticCredentialForTest')
    class Opener:
        def open(self,request,timeout):
            assert request.method=='HEAD' and request.host=='huggingface.co'
            raise auth.urllib.error.HTTPError(request.full_url,403,'denied',{},None)
    monkeypatch.setattr(auth.urllib.request,'build_opener',lambda *a:Opener())
    with pytest.raises(RuntimeError,match='access denied'):auth.verify_access()
