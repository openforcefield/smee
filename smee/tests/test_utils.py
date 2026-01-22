import openff.interchange.models
import openff.toolkit
import pytest
import torch

import smee
import smee.utils


def test_find_exclusions_simple():
    molecule = openff.toolkit.Molecule()
    for _ in range(6):
        molecule.add_atom(6, 0, False)
    for i in range(5):
        molecule.add_bond(i, i + 1, 1, False)

    exclusions = smee.utils.find_exclusions(molecule.to_topology())
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (0, 3): "scale_14",
        (0, 4): "scale_15",
        (1, 2): "scale_12",
        (1, 3): "scale_13",
        (1, 4): "scale_14",
        (1, 5): "scale_15",
        (2, 3): "scale_12",
        (2, 4): "scale_13",
        (2, 5): "scale_14",
        (3, 4): "scale_12",
        (3, 5): "scale_13",
        (4, 5): "scale_12",
    }


def test_find_exclusions_rings():
    molecule = openff.toolkit.Molecule()
    for _ in range(8):
        molecule.add_atom(6, 0, False)

    # para substituted 6-membered ring
    molecule.add_bond(0, 1, 1, False)
    for i in range(6):
        molecule.add_bond(i + 1, (i + 1) % 6 + 1, 1, False)
    molecule.add_bond(2, 7, 1, False)

    exclusions = smee.utils.find_exclusions(molecule.to_topology())
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (0, 6): "scale_13",
        (0, 3): "scale_14",
        (0, 5): "scale_14",
        (0, 7): "scale_14",
        (0, 4): "scale_15",
        (1, 2): "scale_12",
        (1, 6): "scale_12",
        (1, 3): "scale_13",
        (1, 5): "scale_13",
        (1, 7): "scale_13",
        (1, 4): "scale_14",
        (2, 3): "scale_12",
        (2, 6): "scale_13",
        (2, 4): "scale_13",
        (2, 5): "scale_14",
        (2, 7): "scale_12",
        (3, 4): "scale_12",
        (3, 5): "scale_13",
        (3, 6): "scale_14",
        (3, 7): "scale_13",
        (4, 5): "scale_12",
        (4, 6): "scale_13",
        (4, 7): "scale_14",
        (5, 6): "scale_12",
        (5, 7): "scale_15",
        (6, 7): "scale_14",
    }


def test_find_exclusions_dimer():
    molecule = openff.toolkit.Molecule()
    for _ in range(3):
        molecule.add_atom(6, 0, False)

    molecule.add_bond(0, 1, 1, False)
    molecule.add_bond(1, 2, 1, False)

    topology = openff.toolkit.Topology()
    topology.add_molecule(molecule)
    topology.add_molecule(molecule)

    exclusions = smee.utils.find_exclusions(topology)
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (1, 2): "scale_12",
        (3, 4): "scale_12",
        (3, 5): "scale_13",
        (4, 5): "scale_12",
    }


def test_find_exclusions_v_sites():
    molecule = openff.toolkit.Molecule()
    for _ in range(4):
        molecule.add_atom(6, 0, False)
    for i in range(3):
        molecule.add_bond(i, i + 1, 1, False)

    v_site_keys = [
        openff.interchange.models.VirtualSiteKey(
            orientation_atom_indices=(0, 1, 2),
            type="MonovalentLonePair",
            match="once",
            name="EP",
        ),
        openff.interchange.models.VirtualSiteKey(
            orientation_atom_indices=(3, 2, 1),
            type="MonovalentLonePair",
            match="once",
            name="EP",
        ),
    ]

    v_sites = smee.VSiteMap(
        keys=v_site_keys,
        key_to_idx={v_site_keys[0]: 4, v_site_keys[1]: 5},
        parameter_idxs=torch.zeros((2, 1)),
    )

    exclusions = smee.utils.find_exclusions(molecule.to_topology(), v_sites)
    assert exclusions == {
        (0, 1): "scale_12",
        (0, 2): "scale_13",
        (0, 3): "scale_14",
        (0, 4): "scale_12",
        (0, 5): "scale_14",
        (1, 2): "scale_12",
        (1, 3): "scale_13",
        (1, 4): "scale_12",
        (1, 5): "scale_13",
        (2, 3): "scale_12",
        (2, 4): "scale_13",
        (2, 5): "scale_12",
        (3, 4): "scale_14",
        (3, 5): "scale_12",
        (4, 5): "scale_14",
    }


def test_ones_like():
    expected_size = (4, 5)
    expected_type = torch.float16

    other = torch.tensor([1, 2, 3], dtype=expected_type, device="cpu")
    tensor = smee.utils.ones_like(expected_size, other)

    assert tensor.dtype == expected_type
    assert tensor.shape == expected_size
    assert torch.allclose(tensor, torch.tensor(1.0, dtype=expected_type))


def test_zeros_like():
    expected_size = (4, 5)
    expected_type = torch.float16

    other = torch.tensor([1, 2, 3], dtype=expected_type, device="cpu")
    tensor = smee.utils.zeros_like(expected_size, other)

    assert tensor.dtype == expected_type
    assert tensor.shape == expected_size
    assert torch.allclose(tensor, torch.tensor(0.0, dtype=expected_type))


def test_tensor_like():
    expected_type = torch.float16
    expected_data = [[3.0], [2.0], [1.0]]

    other = torch.tensor([], dtype=expected_type, device="cpu")
    tensor = smee.utils.tensor_like(expected_data, other)

    assert tensor.dtype == expected_type
    assert torch.allclose(tensor, torch.tensor(expected_data, dtype=expected_type))


def test_tensor_like_copy():
    expected_type = torch.float16
    expected_data = torch.tensor([[3.0], [2.0], [1.0]], requires_grad=True)

    other = torch.tensor(expected_data, dtype=expected_type, device="cpu")
    tensor = smee.utils.tensor_like(expected_data, other)

    assert tensor.requires_grad is False
    assert tensor.dtype == expected_type
    assert torch.allclose(tensor, torch.tensor(expected_data, dtype=expected_type))


def test_arange_like():
    expected_type = torch.int8
    expected_data = [0, 1, 2, 3]

    other = torch.tensor([], dtype=expected_type, device="cpu")
    tensor = smee.utils.arange_like(4, other)

    assert tensor.dtype == expected_type
    assert torch.allclose(tensor, torch.tensor(expected_data, dtype=expected_type))


@pytest.mark.parametrize(
    "a, b, dim, keepdim",
    [
        (torch.tensor([1.0, 2.0, 3.0]), None, 0, False),
        (torch.tensor([1.0, 2.0, 3.0]), None, 0, True),
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor(0.0), 0, False),
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor(2.0), 0, False),
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0]), 0, False),
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor(2.0), 0, True),
        (torch.tensor([1.0, 2.0, 3.0]), torch.tensor([3.0, 2.0, 1.0]), 0, True),
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
            0,
            False,
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
            1,
            False,
        ),
        (
            torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            torch.tensor([[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]),
            1,
            True,
        ),
        (torch.tensor(-torch.inf), torch.tensor(1.0), 0, True),
    ],
)
def test_logsumexp(a, b, dim, keepdim):
    from scipy.special import logsumexp

    actual = smee.utils.logsumexp(a, dim, keepdim, b)
    expected = torch.tensor(
        logsumexp(a.numpy(), dim, b if b is None else b.numpy(), keepdim)
    )

    # scipy.logsumexp returns shape (1,) for 0-d inputs with keepdim=True
    # but we expect shape () for 0-d inputs regardless of keepdim
    if a.ndim == 0 and expected.ndim == 1:
        expected = expected.squeeze()

    assert actual.shape == expected.shape
    assert torch.allclose(actual.double(), expected.double())


def test_logsumexp_with_sign():
    from scipy.special import logsumexp

    a = torch.tensor([1.0, 2.0, 3.0])
    b = torch.tensor(-2.0)

    actual, actual_sign = smee.utils.logsumexp(a, -1, True, b, return_sign=True)
    expected, expected_sign = torch.tensor(
        logsumexp(a.numpy(), -1, b.numpy(), True, return_sign=True)
    )

    assert actual.shape == expected.shape
    assert torch.allclose(actual.double(), expected.double())

    assert actual_sign.shape == expected_sign.shape
    assert torch.allclose(actual_sign.double(), expected_sign.double())


@pytest.mark.parametrize("n", [7499, 7500, 7501])
def test_to_upper_tri_idx(n):
    i, j = torch.triu_indices(n, n, 1)
    expected_idxs = torch.arange(len(i))

    idxs = smee.utils.to_upper_tri_idx(i, j, n)

    assert idxs.shape == expected_idxs.shape
    assert (idxs == expected_idxs).all()


def test_geometric_mean():
    a = torch.tensor(4.0, requires_grad=True).double()
    b = torch.tensor(9.0, requires_grad=True).double()

    assert torch.autograd.gradcheck(
        smee.utils.geometric_mean, (a, b), check_backward_ad=True, check_forward_ad=True
    )

    assert torch.isclose(smee.utils.geometric_mean(a, b), torch.tensor(6.0).double())


@pytest.mark.parametrize(
    "a, b, expected_grad_a, expected_grad_b",
    [
        (0.0, 0.0, 0.0, 0.0),
        (3.0, 0.0, 0.0, 3.0 / (2.0 * smee.utils.EPSILON)),
        (0.0, 4.0, 4.0 / (2.0 * smee.utils.EPSILON), 0.0),
    ],
)
def test_geometric_mean_zero(a, b, expected_grad_a, expected_grad_b):
    a = torch.tensor(a, requires_grad=True)
    b = torch.tensor(b, requires_grad=True)

    v = smee.utils.geometric_mean(a, b)
    v.backward()

    expected_grad_a = torch.tensor(expected_grad_a)
    expected_grad_b = torch.tensor(expected_grad_b)

    assert a.grad.shape == expected_grad_a.shape
    assert torch.allclose(a.grad, expected_grad_a)

    assert b.grad.shape == expected_grad_b.shape
    assert torch.allclose(b.grad, expected_grad_b)
