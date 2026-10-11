from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from config import Config

ET = ZoneInfo('America/New_York')
FEATURES = ['momentum_6h','momentum_24h','momentum_72h','rv_6h','rv_24h','rv_7d','volume_zscore','jump_factor','mark_iv','bid_iv','ask_iv','iv_minus_rv','moneyness','log_moneyness','relative_spread','delta','gamma','vega','theta','strike','is_call','hours_to_expiry']

def utc(t):
    t = pd.Timestamp(t)
    if t.tzinfo is None:
        raise ValueError('Timestamp must be timezone-aware')
    return t.tz_convert('UTC')

def weekend_id(t):
    local = utc(t).tz_convert(ET)
    return (local - pd.Timedelta(days=(local.weekday()-5)%7)).date().isoformat()

def saturday(t):
    return utc(t).tz_convert(ET).weekday() == 5

def payoff(s, k, kind):
    return max(s-k,0) if kind == 'CALL' else max(k-s,0)

def bs(s,k,h,iv,kind):
    tau = max(h/8760, 1e-10)
    d1 = (np.log(s/k)+iv*iv*tau/2)/(iv*np.sqrt(tau))
    d2 = d1-iv*np.sqrt(tau)
    call = s*norm.cdf(d1)-k*norm.cdf(d2)
    price = call if kind == 'CALL' else call-s+k
    delta = norm.cdf(d1) - (kind == 'PUT')
    gamma = norm.pdf(d1)/(s*iv*np.sqrt(tau))
    vega = s*norm.pdf(d1)*np.sqrt(tau)
    theta = -s*norm.pdf(d1)*iv/(2*np.sqrt(tau)*365)
    return max(price,0),delta,gamma,vega,theta

def market_factors(spot,t):
    h = spot.loc[spot.timestamp <= utc(t)].sort_values('timestamp')
    if len(h)<169 or (utc(t)-h.timestamp.iloc[-1]).total_seconds()>3600:
        raise ValueError('Insufficient or stale underlying history')
    p = h.spot.to_numpy(); r = np.diff(np.log(p))
    out = {f'momentum_{n}h':p[-1]/p[-n-1]-1 for n in (6,24,72)}
    out.update({f'rv_{name}':np.std(r[-n:],ddof=1)*np.sqrt(8760) for name,n in [('6h',6),('24h',24),('7d',168)]})
    v=h.volume.iloc[-168:]
    out['volume_zscore']=(v.iloc[-1]-v.mean())/max(v.std(),1e-10)
    out['jump_factor']=abs(out['momentum_24h'])/max(out['rv_24h']*np.sqrt(24/8760),1e-8)
    return out,float(p[-1])

def candidates(chain,spot,t,cfg=Config()):
    t=utc(t)
    rows=chain.loc[chain.timestamp<=t].sort_values('timestamp').groupby('symbol').tail(1).copy()
    if rows.empty: return rows, []
    mf,s=market_factors(spot,t)
    audit=[]; accepted=[]
    for _,r in rows.iterrows():
        reason=None
        required=['bid','ask','bid_size','ask_size','mark_iv','bid_iv','ask_iv','delta','gamma','vega','theta','strike']
        if not np.isfinite(r[required].astype(float)).all(): reason='missing/nonfinite quote'
        elif r.unit != 1 or r.min_qty <= 0: reason='unsupported multiplier or lot size'
        elif not (0<r.bid<r.ask and r.bid_size>0 and r.ask_size>0): reason='invalid book'
        elif (t-r.timestamp).total_seconds()>cfg.max_quote_age_seconds: reason='stale quote'
        elif r.expiry<=t or r.expiry.tz_convert(ET).weekday()!=6: reason='not future Sunday ET expiry'
        if reason:
            audit.append({'time':str(t),'symbol':r.symbol,'reason':reason}); continue
        d=r.to_dict(); d.update(mf)
        d.update(spot=s,weekend_id=weekend_id(t),decision_time=t,expiry_utc=r.expiry,expiry_et=r.expiry.tz_convert(ET),hours_to_expiry=(r.expiry-t).total_seconds()/3600,moneyness=s/r.strike,log_moneyness=np.log(s/r.strike),relative_spread=(r.ask-r.bid)/((r.ask+r.bid)/2),iv_minus_rv=r.mark_iv-mf['rv_24h'],is_call=float(r.option_type=='CALL'))
        accepted.append(d)
    return pd.DataFrame(accepted),audit

def costs(bid, exit_price, s, cfg=Config()):
    fees=min(cfg.fee_rate*s,0.1*bid)+min(cfg.fee_rate*s,0.1*exit_price)
    return fees,2*cfg.slippage

def label(row,spot,cfg=Config()):
    # Settlement requires an exact exchange-index observation, never a future nearest price.
    match=spot.loc[spot.timestamp==row.expiry]
    if len(match)!=1: raise ValueError('Missing exact settlement index')
    settlement=payoff(float(match.spot.iloc[0]),row.strike,row.option_type)
    fees,slip=costs(row.bid,settlement,row.spot,cfg)
    return row.bid-settlement-fees-slip,settlement,fees,slip

class FactorModel:
    def __init__(self,kind='ridge'):
        self.scaler=StandardScaler(); self.model=Ridge(alpha=10) if kind=='ridge' else LinearRegression()
    def fit(self,train,t):
        train=train.loc[(train.weekend_id<weekend_id(t)) & (train.label_available_at<utc(t)) & train.y.notna()]
        if train.empty: raise ValueError('No matured training labels')
        x=train[FEATURES].to_numpy(float); y=train.y.to_numpy(float)
        z=self.scaler.fit_transform(x); self.model.fit(z,y)
        self.beta=self.model.coef_/self.scaler.scale_
        self.intercept=float(self.model.intercept_-self.scaler.mean_@self.beta)
        self.sigma=float(np.std(y-self.model.predict(z)))
        self.n=len(train); self.train_ids=sorted(train.weekend_id.unique())
        return self
    def predict(self,rows):
        return self.intercept+rows[FEATURES].to_numpy(float)@self.beta

def optimize(rows,alpha,cfg=Config()):
    n=len(rows)
    if not n: return None
    g=rows[['delta','gamma','vega','theta']].to_numpy(float)
    margin=np.maximum(rows.spot.to_numpy()*0.2,rows.bid.to_numpy())
    risk=np.maximum(rows.spot.to_numpy()*abs(g[:,0])*0.1 + rows.spot.to_numpy()**2*abs(g[:,1])*0.005 + abs(g[:,2])*0.25,1)
    # q is short quantity; published signed option weights are w = -q.
    constraints=[{'type':'ineq','fun':lambda q:cfg.gross_limit-q.sum()}, {'type':'ineq','fun':lambda q:cfg.capital_limit-margin@q}]
    for j,limit in enumerate([cfg.delta_limit,cfg.gamma_limit,cfg.vega_limit]):
        constraints.append({'type':'ineq','fun':lambda q,j=j,limit=limit:limit-abs(g[:,j]@q)})
    res=minimize(lambda q:-alpha@q+cfg.risk_penalty*np.sum((risk*q)**2),np.zeros(n),method='SLSQP',bounds=[(0,min(cfg.max_position,float(r.bid_size))) for r in rows.itertuples()],constraints=constraints,options={'maxiter':500,'ftol':1e-9})
    if not res.success: return None
    q=res.x
    if any(c['fun'](q)<-1e-6 for c in constraints): return None
    expected=float(alpha@q); risk_measure=float(np.linalg.norm(risk*q))
    return dict(weights=-q,greeks=-q@g,expected=expected,risk=risk_measure,risk_per_unit=risk,score=expected/max(risk_measure,1e-9),constraint_slacks=[float(c['fun'](q)) for c in constraints])
