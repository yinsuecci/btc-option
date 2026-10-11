import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
from html.parser import HTMLParser
import pandas as pd

class Links(HTMLParser):
    def __init__(self): super().__init__(); self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            if 'files/' in href or 'Deribit' in href or 'instrument_name' in href: self.links.append(href)

def main():
    p=Path('results/source_search'); evidence={}
    with urlopen('https://www.kaggle.com/api/v1/datasets/download/hsergeyfrolov/deribit-btc-options-information',timeout=30) as response: blob=response.read(3000000)
    (p/'kaggle_deribit.zip').write_bytes(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        evidence['kaggle_files']=archive.namelist(); frames=[]
        for name in archive.namelist():
            if name.endswith('.csv'):
                with archive.open(name) as file: frame=pd.read_csv(file)
                bounds={}
                if 'timestamp' in frame:
                    times=pd.to_datetime(frame.timestamp,unit='ms',utc=True)
                    bounds={'first_timestamp':str(times.min()),'last_timestamp':str(times.max()),'distinct_utc_days':int(times.dt.date.nunique())}
                frames.append({'file':name,'rows':len(frame),'columns':list(frame.columns),**bounds,'head':frame.head(2).astype(str).fillna('MISSING').to_dict('records')})
        evidence['kaggle_schema']=frames
    try:
        with urlopen('https://research-repository.uwa.edu.au/en/datasets/deribits-options-data/',timeout=30) as response: page=response.read().decode()
        parser=Links(); parser.feed(page); evidence['uwa_links']=parser.links
    except HTTPError as error:
        evidence['uwa_access']={'http_status':error.code,'status':'ACCESS_BLOCKED'}
    (p/'open_dataset_schema.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    print(json.dumps(evidence,indent=2))
if __name__=='__main__': main()
