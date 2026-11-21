"""Exercise all dashboard model/split combinations via a Streamlit-only stub.

This is a pure source/data-flow test; it is not a browser-render test.
"""
from pathlib import Path
import runpy
import sys
from types import ModuleType
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'Sources'))
from run_integrated_synthetic_qa import main as make_fixture

@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    base=tmp_path_factory.mktemp('fix3_ui_matrix')
    out,raw=base/'run',base/'raw'
    assert make_fixture(['--out-dir',str(out),'--raw-dir',str(raw)])==0
    return out,raw


@pytest.mark.parametrize('variant',['three_features','five_features'])
@pytest.mark.parametrize('split',['validation','test'])
def test_ui_all_variant_partition_paths(synthetic,monkeypatch,variant,split):
    results,raw=synthetic
    monkeypatch.setenv('CROSSWATCH_RESULTS',str(results))
    monkeypatch.setenv('CROSSWATCH_RAW',str(raw))
    class StreamlitStub(ModuleType):
        def __init__(self):
            super().__init__('streamlit')
            self.visited=[]
            self.tab_names=[]
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getattr__(self,name):
            if name=='stop':
                def fail():raise RuntimeError('Dashboard stopped unexpectedly: '+str(self.visited))
                return fail
            def capture(*args,**kwargs):
                self.visited.append(name)
                if name=='columns':return [self for _ in range(args[0])]
                if name=='tabs':
                    self.tab_names=list(args[0]);return [self for _ in args[0]]
                if name=='selectbox':
                    label=args[0]
                    if label=='Forest feature set':return variant
                    if label=='Evaluation partition':return split
                    return args[1][kwargs.get('index',0)]
            return capture
    stub=StreamlitStub()
    monkeypatch.setitem(sys.modules,'streamlit',stub)
    runpy.run_path(str(ROOT/'Sources'/'streamlit_app.py'),run_name=f'fix3_matrix_{split}_{variant}')
    assert len(stub.tab_names)==5
    assert 'error' not in stub.visited
    assert stub.visited.count('line_chart')==3
    assert 'pyplot' in stub.visited
    assert stub.visited.count('dataframe')>=6
