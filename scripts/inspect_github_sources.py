"""Audit public GitHub data inventories without running repository code."""
import csv
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPOS = [
    'bottama/Deribit-Option-Data',
    'Ronak-Mahajan/neural-options-lab',
    'Kirill-Miroshnichenko/deribit-options-collector',
    'schepal/deribit_data_collector',
    'nostoz/deribit_volatility_download_and_visualize',
    'RiveChen/deribit-historical-data',
    'gptcompany/deribit-data-downloader',
    'BobbyAxerol/trading-historical-data',
    'bykarantelicom/crypto-datasets',
    'FinkBig/THESIS_deribit_probability_analyzer',
    'VijitSingh97/quant',
    'QuantLet/hedging_cc',
]


def get(url, cap=8_000_000):
    request = Request(url, headers={'User-Agent': 'BTC-weekend-research',
                                  'Accept': 'application/vnd.github+json'})
    with urlopen(request, timeout=45) as response:
        body = response.read(cap + 1)
    return body[:cap], len(body) > cap


def api(path):
    body, truncated = get('https://api.github.com/' + path)
    if truncated:
        raise ValueError('API body exceeded audit cap')
    return json.loads(body)


def main():
    output = Path('results/source_search/github')
    output.mkdir(parents=True, exist_ok=True)
    audits = []
    for repo in REPOS:
        record = {'repository': 'https://github.com/' + repo,
                  'audited_at': datetime.now(timezone.utc).isoformat()}
        try:
            metadata = api('repos/' + repo)
            branches = [metadata['default_branch']]
            if repo == 'Ronak-Mahajan/neural-options-lab':
                branches.append('surfaces')
            record['branches'] = []
            for branch in branches:
                tree = api(f'repos/{repo}/git/trees/{branch}?recursive=1')
                blobs = [r for r in tree.get('tree', []) if r['type'] == 'blob']
                data = [r for r in blobs if re.search(
                    r'\.(csv|parquet|jsonl|gz|zip|sqlite|db)$', r['path'], re.I)]
                dates = sorted(set(m.group(0) for r in data for m in
                                   re.finditer(r'20\d{2}[-_]?\d{2}[-_]?\d{2}', r['path'])))
                info = {'branch': branch, 'tree_truncated': tree.get('truncated'),
                        'blob_count': len(blobs), 'data_file_count': len(data),
                        'filename_date_min': dates[0] if dates else None,
                        'filename_date_max': dates[-1] if dates else None,
                        'data_files': data}
                # Inspect selected raw CSVs, retaining the cap/completeness caveat.
                candidates = [r for r in data if r['path'].endswith('.csv')]
                candidates.sort(key=lambda r: ('btc' not in r['path'].lower(), r['path']))
                info['samples'] = []
                for item in candidates[:2]:
                    url = f'https://raw.githubusercontent.com/{repo}/{branch}/' + quote(item['path'])
                    raw, capped = get(url)
                    sample = {'path': item['path'], 'download_capped': capped,
                              'bytes_inspected': len(raw)}
                    decoded = raw.decode('utf-8-sig', errors='replace')
                    if decoded.startswith('version https://git-lfs.github.com/spec'):
                        sample['lfs_pointer'] = decoded
                    else:
                        rows = list(csv.DictReader(io.StringIO(decoded)))
                        sample['rows_inspected'] = len(rows)
                        sample['columns'] = list(rows[0]) if rows else []
                        sample['first_row'] = rows[0] if rows else None
                        sample['last_row'] = rows[-1] if rows else None
                        for key in sample['columns']:
                            if key.lower() in ('timestamp', 'time', 'date', 'datetime', 'timestamp_ms'):
                                values = [r.get(key) for r in rows if r.get(key)]
                                if values:
                                    try:
                                        numbers = [float(v) for v in values]
                                        bounds = [min(numbers), max(numbers)]
                                        unit = 1e3 if bounds[-1] > 1e12 else 1
                                        sample[key + '_utc_bounds'] = [datetime.fromtimestamp(v / unit, timezone.utc).isoformat() for v in bounds]
                                    except (ValueError, OverflowError, OSError):
                                        sample[key + '_string_bounds'] = [min(values), max(values)]
                    info['samples'].append(sample)
                record['branches'].append(info)
            readme = api('repos/' + repo + '/readme')
            body, capped = get(readme['download_url'])
            (output / (repo.replace('/', '__') + '_README.txt')).write_bytes(body)
        except (HTTPError, ValueError, TimeoutError) as error:
            record['error'] = str(error)
        audits.append(record)
        (output / 'inventory.json').write_text(json.dumps(audits, indent=2), encoding='utf-8')
        print(json.dumps({'repo': repo, 'error': record.get('error'), 'branches': [
            {k: v for k, v in b.items() if k not in ('data_files', 'samples')}
            for b in record.get('branches', [])]}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
