import argparse
import json
from src.data import load
from src.research import run
from config import Config

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--data'); parser.add_argument('--output',default='results'); parser.add_argument('--threshold',type=float,default=.02); parser.add_argument('--rolling-weekends',type=int); parser.add_argument('--decision-frequency',type=int,default=1)
    a=parser.parse_args()
    if a.decision_frequency<1: parser.error('decision-frequency must be positive')
    spot,chain,mode=load(a.data)
    _,metrics=run(spot,chain,mode,Config(threshold=a.threshold,rolling_weekends=a.rolling_weekends,decision_frequency=a.decision_frequency),a.output)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
