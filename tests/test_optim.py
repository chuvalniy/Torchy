import numpy as np

from torchy.optim import MomentumSGD, RMSProp, Adam
from tests.utils import rel_error
from torchy.value import Value


def test_sgd_momentum():
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    param = Value(w)
    param.grad = dw
    params = {"W": param}

    sgd_momentum = MomentumSGD(params, lr=1e-3)
    sgd_momentum._velocities = {"W": v}

    sgd_momentum.step()

    expected_next_w = np.asarray([
        [0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
        [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
        [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
        [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096]])
    expected_velocity = np.asarray([
        [0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158],
        [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105],
        [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053],
        [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096]])

    assert rel_error(sgd_momentum._params['W'].data, expected_next_w) <= 1e-8
    assert rel_error(expected_velocity, sgd_momentum._velocities['W']) <= 1e-8


def test_rmsprop():
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    cache = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    param = Value(w)
    param.grad = dw
    params = {"W": param}

    optimizer = RMSProp(params, lr=1e-2)
    optimizer._accumulated = {"W": cache}
    optimizer.step()

    expected_next_w = np.asarray([
        [-0.39223849, -0.34037513, -0.28849239, -0.23659121, -0.18467247],
        [-0.132737, -0.08078555, -0.02881884, 0.02316247, 0.07515774],
        [0.12716641, 0.17918792, 0.23122175, 0.28326742, 0.33532447],
        [0.38739248, 0.43947102, 0.49155973, 0.54365823, 0.59576619]])
    expected_cache = np.asarray([
        [0.5976, 0.6126277, 0.6277108, 0.64284931, 0.65804321],
        [0.67329252, 0.68859723, 0.70395734, 0.71937285, 0.73484377],
        [0.75037008, 0.7659518, 0.78158892, 0.79728144, 0.81302936],
        [0.82883269, 0.84469141, 0.86060554, 0.87657507, 0.8926]])

    # You should see relative errors around e-7 or less
    assert rel_error(expected_next_w, optimizer._params['W'].data) <= 1e-7
    assert rel_error(expected_cache, optimizer._accumulated['W']) <= 1e-7


def test_adam():
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    m = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)
    v = np.linspace(0.7, 0.5, num=N * D).reshape(N, D)

    param = Value(w)
    param.grad = dw
    params = {"W": param}

    optimizer = Adam(params, lr=1e-2, t=5)
    optimizer._velocities = {"W": m}
    optimizer._accumulated = {"W": v}
    optimizer.step()

    expected_next_w = np.asarray([
        [-0.40094747, -0.34836187, -0.29577703, -0.24319299, -0.19060977],
        [-0.1380274, -0.08544591, -0.03286534, 0.01971428, 0.0722929],
        [0.1248705, 0.17744702, 0.23002243, 0.28259667, 0.33516969],
        [0.38774145, 0.44031188, 0.49288093, 0.54544852, 0.59801459]])
    expected_v = np.asarray([
        [0.69966, 0.68908382, 0.67851319, 0.66794809, 0.65738853, ],
        [0.64683452, 0.63628604, 0.6257431, 0.61520571, 0.60467385, ],
        [0.59414753, 0.58362676, 0.57311152, 0.56260183, 0.55209767, ],
        [0.54159906, 0.53110598, 0.52061845, 0.51013645, 0.49966, ]])
    expected_m = np.asarray([
        [0.48, 0.49947368, 0.51894737, 0.53842105, 0.55789474],
        [0.57736842, 0.59684211, 0.61631579, 0.63578947, 0.65526316],
        [0.67473684, 0.69421053, 0.71368421, 0.73315789, 0.75263158],
        [0.77210526, 0.79157895, 0.81105263, 0.83052632, 0.85]])

    assert rel_error(expected_next_w, optimizer._params['W'].data) <= 1e-6
    assert rel_error(expected_v, optimizer._accumulated['W']) <= 1e-8
    assert rel_error(expected_m, optimizer._velocities['W']) <= 1e-8
