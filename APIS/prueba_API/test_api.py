import numpy as np
from functions import Tiro_parabolico

def test_tiro_parabolico_largo():
    x,y,t = Tiro_parabolico(45,50)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert x.shape[0] == y.shape[0] == t.shape[0]

def test_tiro_vertical():
    x,y,t = Tiro_parabolico(90,50)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert np.allclose(x,0.0,atol=1e-7)

def test_tiro_parabolico_cero():
    x,y,t = Tiro_parabolico(0,50)
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(t, np.ndarray)
    assert np.allclose(y,0.0,atol=1e-7)
