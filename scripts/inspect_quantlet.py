import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.request import urlopen

def main():
    p=Path('results/source_search')
    with urlopen('https://api.github.com/repos/QuantLet/hedging_cc/git/trees/master?recursive=1',timeout=30) as response: tree=json.load(response)
    groups=defaultdict(list)
    for r in tree.get('tree',[]):
        if r['path'].startswith('_input/') and r['path'].endswith('.csv'):
            key='/'.join(r['path'].split('/')[:2]); groups[key].append(r)
    info={'repository':'https://github.com/QuantLet/hedging_cc','tree_truncated':tree.get('truncated'),'groups':[]}
    for key,rows in groups.items():
        dates=sorted(m.group(0) for r in rows if (m:=re.search(r'(?<!\d)20\d{6}(?!\d)',r['path'])))
        group={'prefix':key,'file_count':len(rows),'first_date':dates[0] if dates else None,'last_date':dates[-1] if dates else None}
        first=rows[0]['path']
        with urlopen('https://raw.githubusercontent.com/QuantLet/hedging_cc/master/'+first,timeout=30) as response: sample=response.read(4096).decode()
        group['sample_path']=first; group['sample_head']=sample.splitlines()[:2]; info['groups'].append(group)
    (p/'quantlet_audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(json.dumps(info,indent=2))
if __name__=='__main__': main()
