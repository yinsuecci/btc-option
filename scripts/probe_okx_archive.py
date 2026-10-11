import json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError,URLError
import pandas as pd

def main():
    records=[]
    for date in ['2023-04-01','2024-01-06','2025-04-05']:
        t=int(pd.Timestamp(date,tz='UTC').timestamp()*1000)
        params={'module':'4','instType':'OPTION','dateAggrType':'daily','begin':str(t),'end':str(t),'instFamilyList':'BTC-USD'}
        url='https://www.okx.com/api/v5/public/market-data-history?'+urlencode(params)
        record={'date':date,'url':url}
        try:
            with urlopen(url,timeout=30) as response: record['response']=json.load(response); record['http_status']=response.status
        except HTTPError as error: record['http_status']=error.code
        except (URLError,TimeoutError) as error: record['error']=str(error)
        records.append(record)
    Path('results/source_search/okx_archive_probes.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
if __name__=='__main__': main()
