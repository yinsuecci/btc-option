import io
from scripts import audit_real_history as audit

def test_archive_listing_pagination(monkeypatch):
    calls=[]
    pages=[b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Contents><Key>first.zip</Key></Contents><NextContinuationToken>next</NextContinuationToken></ListBucketResult>',b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/"><Contents><Key>second.zip</Key></Contents><CommonPrefixes><Prefix>data/option/daily/EOHSummary/</Prefix></CommonPrefixes></ListBucketResult>']
    def fake(url,timeout):
        calls.append(url)
        return io.BytesIO(pages[len(calls)-1])
    monkeypatch.setattr(audit,'urlopen',fake)
    keys,prefixes=audit.listing('data/option/daily/','/')
    assert keys==['first.zip','second.zip']
    assert prefixes==['data/option/daily/EOHSummary/']
    assert 'continuation-token=next' in calls[1]
