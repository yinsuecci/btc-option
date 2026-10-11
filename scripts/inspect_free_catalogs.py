"""Read public dataset catalogs; do not mistake repository size for history."""
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TARGETS = {
    'kaggle_bitcoin_options': 'https://www.kaggle.com/api/v1/datasets/list?search=bitcoin%20options',
    'kaggle_crypto_options': 'https://www.kaggle.com/api/v1/datasets/list?search=crypto%20options',
    'hf_deribit_catalog': 'https://huggingface.co/api/datasets?search=deribit&limit=30',
    'hf_azulcoder_metadata': 'https://huggingface.co/api/datasets/azulcoder/btc-quant-ticks',
    'hf_azulcoder_files': 'https://huggingface.co/api/datasets/azulcoder/btc-quant-ticks/tree/main/data?limit=1000',
    'mendeley_short_sample': 'https://data.mendeley.com/datasets/hfd4b8zfw6/1',
    'uforia_landing': 'https://uforia.polynomial.fi/datasets?dataset=option_chain_snapshot',
}


def main():
    output = Path('results/free_source_search/catalogs')
    output.mkdir(parents=True, exist_ok=True)
    evidence = []
    for name, url in TARGETS.items():
        row = {'name': name, 'url': url}
        try:
            with urlopen(Request(url, headers={'User-Agent': 'BTC-options-research'}), timeout=30) as response:
                body = response.read(4_000_001)
                row['http_status'] = response.status
            row['body_capped'] = len(body) > 4_000_000
            (output / (name + '.txt')).write_bytes(body[:4_000_000])
            if not row['body_capped']:
                try:
                    data = json.loads(body)
                    if name.startswith('kaggle'):
                        row['datasets'] = [{key: r.get(key) for key in
                                            ['ref', 'title', 'totalBytes', 'lastUpdated']} for r in data]
                    elif name == 'hf_deribit_catalog':
                        row['datasets'] = [r['id'] for r in data]
                    elif name == 'hf_azulcoder_files':
                        dates = sorted(set(m.group(0) for r in data for m in
                                           re.finditer(r'20\d{2}-\d{2}-\d{2}', r.get('path', ''))))
                        row.update(listed_dates=len(dates),
                                   first_listed_date=dates[0] if dates else None,
                                   last_listed_date=dates[-1] if dates else None,
                                   pagination_not_exhausted=True)
                    elif name == 'hf_azulcoder_metadata':
                        row.update(created_at=data.get('createdAt'), sha=data.get('sha'))
                except (json.JSONDecodeError, TypeError):
                    pass
        except (HTTPError, URLError, TimeoutError) as error:
            row['error'] = str(error)
        evidence.append(row)
        (output / 'catalog_audit.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
