from stms import stms
import numpy as np

def test_stms_init():
    model = stms()
    assert model.n_spline > 0
