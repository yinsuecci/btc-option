"""Publish the validated distribution report without losing the prior alpha report."""
from pathlib import Path
import hashlib
import json

if __name__ == '__main__':
    source=Path('results/center_vol/report.html')
    audit=json.loads(Path('results/center_vol/validation.json').read_text())
    if audit['status']!='PASS':
        raise ValueError('Report must pass artifact audit before publication')
    if hashlib.sha256(source.read_bytes()).hexdigest()!=audit['report_sha256']:
        raise ValueError('Report changed after validation')
    for target in [Path('results/report.html'),Path('results/trade_only/report.html')]:
        backup=target.with_name('report_netalpha.html')
        if target.exists() and not backup.exists():
            backup.write_bytes(target.read_bytes())
        target.write_bytes(source.read_bytes())
    print('Published center-volatility report; previous reports preserved as report_netalpha.html')
