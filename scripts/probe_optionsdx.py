import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request,urlopen

def main():
    records=[]
    for month in ['2022-01','2023-01','2024-09']:
        for freq in ['End of Day','30 Minutes']:
            body=urlencode({'product_id':'1527','attribute_period':month,'attribute_quote-frequency':freq}).encode()
            request=Request('https://www.optionsdx.com/?wc-ajax=get_variation',data=body,headers={'Content-Type':'application/x-www-form-urlencoded','User-Agent':'BTCOptionsResearch/0.1'})
            with urlopen(request,timeout=30) as response: info=json.load(response)
            records.append({'month':month,'frequency':freq,'variation_id':info.get('variation_id') if info else None,'price':info.get('display_price') if info else None,'purchasable':info.get('is_purchasable') if info else None})
    Path('results/source_search/optionsdx_prices.json').write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))
if __name__=='__main__': main()
