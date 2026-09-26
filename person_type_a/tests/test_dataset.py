import pytest

torch = pytest.importorskip("torch")

from person_type_a.dataset import CIFAR10_CLASSES, make_cifar10_scenario


def test_class_count_fits_letters():
    assert len(CIFAR10_CLASSES) == 10
    assert len(CIFAR10_CLASSES) + 1 <= 26


def test_scenario_valid():
    from person_type_a.schema import validate
    sc = make_cifar10_scenario()
    validate(sc)
    q = sc.questions[0]
    assert len(q.effective_options) == 11


def test_scenario_has_descriptions():
    sc = make_cifar10_scenario()
    q = sc.questions[0]
    assert q.criteria is not None
    assert len(q.criteria) >= 10
