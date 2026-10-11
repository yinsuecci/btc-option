import json
from pathlib import Path
from src.data import get_json

def main():
    # Public read-only endpoints; no credentials or order endpoints exist in this project.
    p=Path('data/snapshots'); p.mkdir(parents=True,exist_ok=True)
    raw=get_json('https://eapi.binance.com/eapi/v1/exchangeInfo')
    stamp=raw['serverTime']
    (p/f'{stamp}_exchangeInfo.json').write_text(json.dumps(raw,indent=2))
    for name in ['mark','ticker']:
        (p/f'{stamp}_{name}.json').write_text(json.dumps(get_json(f'https://eapi.binance.com/eapi/v1/{name}'),indent=2))
    print('Saved REAL current metadata/mark/ticker; not historical executable order books.')
if __name__=='__main__': main()
