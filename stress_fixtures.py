"""Synthetic fixtures for deterministic software stress testing ONLY."""
from __future__ import annotations
import numpy as np
import pandas as pd

def alternating_volatility_fixture(n:int=1600,seed:int=8127) -> pd.DataFrame:
    """Artificial recurring low/high VIX blocks with occasional cross-asset shocks.

    Intentionally NOT representative of actual market prices or calibrated risk.
    The alternating regimes exercise both VIX-threshold branches in holdouts.
    """
    rng=np.random.default_rng(seed)
    dates=pd.bdate_range('2018-01-02',periods=n)
    idx=np.arange(n)
    high=(idx//80)%2==1
    returns=rng.normal(0,np.where(high,.012,.006),n)
    vix=np.where(high,28.,14.)+rng.normal(0,2.0,n)
    yield_delta=rng.normal(0,np.where(high,5.,3.),n)
    for day in range(72,n,191):
        returns[day]-=.035
        vix[day]+=7
        yield_delta[day]+=-14
    sp=3000*np.exp(np.cumsum(returns))
    y=3.2+np.cumsum(yield_delta)/100
    ans=pd.DataFrame({'SP500':sp,'VIXCLS':np.clip(vix,5,None),
                      'DGS10':y},index=dates)
    ans.index.name='DATE'
    return ans
