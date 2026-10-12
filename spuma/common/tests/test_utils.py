from spuma.common.utils import derive_seed, safe_name, stable_hash


def test_derive_seed_is_stable():
    assert derive_seed(42, "p1", "solver", 0) == derive_seed(42, "p1", "solver", 0)
    assert derive_seed(42, "p1", "solver", 0) != derive_seed(42, "p1", "solver", 1)
    assert 0 <= derive_seed(1) < 2**32


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert len(stable_hash([1], length=8)) == 8


def test_safe_name():
    assert safe_name("math/test/algebra/3") == "math__test__algebra__3"
