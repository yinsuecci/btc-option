import argparse
from src.data import exchange_chain,load

def main():
    p=argparse.ArgumentParser(); p.add_argument('--real',action='store_true'); a=p.parse_args()
    if a.real:
        chain=exchange_chain(); print('DATA_MODE = REAL (current contract metadata, NOT historical chain)')
    else:
        _,chain,_=load(); chain=chain.drop_duplicates('symbol'); print('DATA_MODE = SYNTHETIC; MOCK contracts only')
    print(chain[['symbol','option_type','strike','expiry']].head(16).to_string(index=False))
if __name__=='__main__': main()
