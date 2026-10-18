import hashlib
import json
import time

import pytest

from raid_experiments.pod_transfer import upload_archive


class Missing(Exception):
    response = {'Error': {'Code': 'NoSuchKey'}}


class S3:
    def __init__(self):
        self.created = 0
        self.parts = {}
        self.calls = []
        self.object = None
        self.fail = None
        self.lost_response = None
        self.creation_timeout = False
        self.completion_timeout = False

    def list_multipart_uploads(self, **kwargs):
        return {'Uploads': [] if not self.created or self.creation_timeout
                else [{'Key': 'archive', 'UploadId': 'owned'}]}

    def create_multipart_upload(self, **kwargs):
        self.created += 1
        if self.creation_timeout:
            raise TimeoutError()
        return {'UploadId': 'owned'}

    def list_parts(self, **kwargs):
        return {'Parts': [{'PartNumber': n, 'Size': len(data),
                          'ETag': '"' + hashlib.md5(data).hexdigest() + '"'}
                         for n, data in self.parts.items()]}

    def upload_part(self, **kwargs):
        n, data = kwargs['PartNumber'], kwargs['Body']
        self.calls.append(n)
        if self.fail == n:
            raise ConnectionError()
        self.parts[n] = data
        if self.lost_response == n:
            raise ConnectionError()
        return {'ETag': '"' + hashlib.md5(data).hexdigest() + '"'}

    def complete_multipart_upload(self, **kwargs):
        self.object = b''.join(self.parts[n] for n in sorted(self.parts))
        if self.completion_timeout:
            raise TimeoutError()

    def head_object(self, **kwargs):
        if self.object is None:
            raise Missing()
        return {'ContentLength': len(self.object)}


def setup(tmp_path):
    path = tmp_path / 'archive'
    path.write_bytes(bytes(range(256)) * 40)
    return path, tmp_path / 'journal.json', int(time.time() * 1000) + 60000


def run(client, source, journal, deadline):
    return upload_archive(client, source, 'existing-volume', 'archive', journal,
                          deadline_ms=deadline, part_bytes=4096, workers=1,
                          attempts=1)


def test_connection_failure_preserves_parts_and_resumes(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    client.fail = 2
    with pytest.raises(ConnectionError):
        run(client, source, journal, deadline)
    assert client.parts[1]
    client.fail = None
    result = run(client, source, journal, deadline)
    assert result['state'] == 'completed'
    assert client.object == source.read_bytes()
    assert client.created == 1 and client.calls.count(1) == 1


def test_lost_part_response_is_reconciled(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    client.lost_response = 2
    result = run(client, source, journal, deadline)
    assert result['state'] == 'completed'
    assert client.calls.count(2) == 1


def test_changed_archive_cannot_reuse_journal(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    run(client, source, journal, deadline)
    source.write_bytes(b'changed')
    with pytest.raises(ValueError, match='identity changed'):
        run(client, source, journal, deadline)


def test_unknown_remote_part_is_rejected(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    client.parts[100] = b'unknown'
    with pytest.raises(ValueError, match='unknown multipart'):
        run(client, source, journal, deadline)


def test_corrupt_part_cannot_resume(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    client.parts[1] = b'wrong'
    with pytest.raises(ValueError, match='checksum mismatch'):
        run(client, source, journal, deadline)


def test_expired_deadline_cannot_create(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    with pytest.raises(TimeoutError):
        run(client, source, journal, 1)
    assert client.created == 0


def test_uncertain_creation_cannot_retry_blindly(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    client.creation_timeout = True
    with pytest.raises(TimeoutError):
        run(client, source, journal, deadline)
    with pytest.raises(RuntimeError, match='never create blindly'):
        run(client, source, journal, deadline)
    assert client.created == 1


def test_completion_timeout_reconciles_and_restart_is_idempotent(tmp_path):
    source, journal, deadline = setup(tmp_path)
    client = S3()
    client.completion_timeout = True
    run(client, source, journal, deadline)
    state = json.loads(journal.read_text())
    state['state'] = 'completing'
    journal.write_text(json.dumps(state))
    calls = len(client.calls)
    assert run(client, source, journal, deadline)['state'] == 'completed'
    assert len(client.calls) == calls and client.created == 1
