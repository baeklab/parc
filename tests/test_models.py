from parc import model
import numpy as np


def test_parc_default():
    parc_default = model.parc()
    assert 17852868 == parc_default.count_params()

    img_size = 64
    in1 = np.random.rand(img_size, img_size, 2)
    in2 = np.random.rand(img_size, img_size, 2)
    in1 = np.expand_dims(in1, 0)
    in2 = np.expand_dims(in2, 0)
    out = parc_default.predict([in1, in2])
    assert out.shape[-1] == 38


def test_parc_adaptive():
    parc_adaptive_ts = model.parc(numTS=5)
    assert 17852868 == parc_adaptive_ts.count_params()
    img_size = 64
    in3 = np.random.rand(
        23, 32, 23, 1, 4, 12, 213, 123, 12, 12, 123, 1, 12, 123, 123, 12
    )
    in1 = np.random.rand(img_size, img_size, 2)
    in2 = np.random.rand(img_size, img_size, 2)
    in1 = np.expand_dims(in1, 0)
    in2 = np.expand_dims(in2, 0)
    out = parc_adaptive_ts.predict([in1, in2])
    assert out.shape[-1] == 10


def test_parc_depth():
    parc_adaptive_depth = model.parc(depth=2)
    assert 8021444 == parc_adaptive_depth.count_params()
