import torch
from nn import layer_norm, std_dev, NNUE
from atlas_chess import Board, Nnue

def test_batch_mean():
    x = torch.tensor([
        [0.25, 0.75, 2],
        [-1, -3, -2]
    ])

    assert (torch.mean(x, dim=1) == torch.tensor([1, -2])).all()

def test_layer_norm():
    x = torch.rand((20, 27), dtype=torch.float32) * 1000
    y = layer_norm(x)

    for i in range(20):
        assert torch.abs(y[i].mean()) < 1e-6
        assert torch.abs(std_dev(y[i], y[i].mean()) - 1.0) < 1e-6

def test_forward_passes_match():
    torch.set_printoptions(sci_mode=False)
    py_nnue = NNUE()

    b = Board()
    n = Nnue()
    n.set_weights(
        w256=py_nnue.l1.weight.detach().numpy(), 
        w64=py_nnue.l2.weight.detach().numpy(),
        w8=py_nnue.l3.weight.detach().numpy(),
        w1=py_nnue.l4.weight.detach().numpy(),
    )
    b.enable_nnue(n)

    b2 = Board("7r/4Nk2/r2p1b2/7P/P1p1P2P/3q4/b1N5/4KR2 w - - 6 63")
    b2.enable_nnue(n)

    rs = [b.forward(), b2.forward()]
    features = torch.tensor([Nnue.features(b), Nnue.features(b2)]).reshape(-1, 40_960 * 2)
    py = py_nnue.forward(features)

    assert abs(rs[0]-py[0]) < 1e-4
    assert abs(rs[1]-py[1]) < 1e-4
