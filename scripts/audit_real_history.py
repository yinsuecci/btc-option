"""Verify public archive coverage without inventing executable options quotes."""
import io
import json
import zipfile
import xml.etree.ElementTree as XML
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import pandas as pd
from src.core import ET

BUCKET='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}

def listing(prefix,delimiter=None):
    keys=[]; prefixes=[]; token=None
    while True:
        params={'list-type':'2','prefix':prefix}
        if delimiter: params['delimiter']=delimiter
        if token: params['continuation-token']=token
        with urlopen(BUCKET+'?'+urlencode(params),timeout=30) as response:
            root=XML.fromstring(response.read())
        keys.extend(n.text for n in root.findall('s:Contents/s:Key',NS))
        prefixes.extend(n.text for n in root.findall('s:CommonPrefixes/s:Prefix',NS))
        token=root.findtext('s:NextContinuationToken',None,NS)
        if not token: return keys,prefixes

def main():
    output=Path('results/real_2022_2026'); output.mkdir(parents=True,exist_ok=True)
    now=pd.Timestamp.now(tz='UTC')
    audit={'checked_at_utc':str(now),'data_mode':'REAL_SOURCE_AUDIT_ONLY','status':'BLOCKED_MISSING_EXECUTABLE_OPTION_HISTORY','beta':None,'omega':None,'datasets':[]}
    _,types=listing('data/option/daily/','/')
    audit['public_archive_types']=types
    for prefix in types:
        keys,_=listing(prefix)
        files=sorted(k for k in keys if k.endswith('.zip') and 'BTC' in k)
        info={'prefix':prefix,'btc_archive_files':len(files),'first_key':files[0] if files else None,'last_key':files[-1] if files else None}
        if files:
            # Inspect actual schema, not a guessed API field list.
            key=files[-1]
            with urlopen('https://data.binance.vision/'+key,timeout=30) as response: blob=response.read()
            (output/(prefix.strip('/').split('/')[-1]+'_sample.zip')).write_bytes(blob)
            with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                member=next(n for n in archive.namelist() if n.endswith('.csv'))
                with archive.open(member) as stream: sample=pd.read_csv(stream,nrows=5)
            info['sample_columns']=list(sample.columns); info['sample_rows']=sample.astype(str).fillna('MISSING').to_dict('records')
        audit['datasets'].append(info)
    rows=[]
    for t in pd.date_range('2022-01-01','2026-12-31',freq='W-SAT',tz=ET):
        finish=t+pd.DateOffset(days=2)
        status='FUTURE' if t.tz_convert('UTC')>now else 'INCOMPLETE_WEEKEND' if finish.tz_convert('UTC')>now else 'MISSING_EXECUTABLE_CHAIN_AND_SETTLEMENT_INDEX'
        rows.append(dict(weekend_id=t.date().isoformat(),saturday_start_utc=str(t.tz_convert('UTC')),weekend_end_utc=str(finish.tz_convert('UTC')),status=status,beta_available=False,omega_available=False))
    pd.DataFrame(rows).to_csv(output/'weekend_coverage.csv',index=False)
    audit['weekend_status_counts']=pd.DataFrame(rows).status.value_counts().to_dict()
    (output/'data_availability.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))
    print('No real beta/omega calculated: historical executable option chains and settlement index are missing.')

if __name__=='__main__': main()
