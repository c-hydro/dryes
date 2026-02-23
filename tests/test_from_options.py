"""Tests for DRYESIndex.from_options factory behavior."""

import pytest

from dryes.indices.dryes_index import DRYESIndex


class _FakeIndex:
    def __init__(self, io_options, index_options, run_options):
        self.io_options = io_options
        self.index_options = index_options
        self.run_options = run_options


class TestDryesIndexFromOptions:
    """Test DRYES index construction from options payloads."""

    def test_from_options_builds_index_without_mutating_input(self, monkeypatch):
        index_options = {"index_name": "spi", "alpha": 1}
        io_options = {"data": object(), "index": object()}
        run_options = {"frequency": "month"}
        index_options_original = index_options.copy()

        monkeypatch.setattr(DRYESIndex, "get_subclass", classmethod(lambda cls, name: _FakeIndex))

        idx = DRYESIndex.from_options(index_options, io_options, run_options)

        assert isinstance(idx, _FakeIndex)
        assert idx.io_options is io_options
        assert idx.run_options is run_options
        assert idx.index_options == {"alpha": 1}
        assert index_options == index_options_original

    def test_from_options_accepts_index_alias_key(self, monkeypatch):
        monkeypatch.setattr(DRYESIndex, "get_subclass", classmethod(lambda cls, name: _FakeIndex))

        idx = DRYESIndex.from_options(
            {"index": "spei", "beta": 2},
            {"data": object(), "index": object()},
            {},
        )

        assert isinstance(idx, _FakeIndex)
        assert idx.index_options == {"beta": 2}

    def test_from_options_rejects_invalid_index_options_type(self):
        with pytest.raises(TypeError, match="'index_options' must be a mapping"):
            DRYESIndex.from_options("bad", {}, {})

    def test_from_options_rejects_invalid_io_options_type(self):
        with pytest.raises(TypeError, match="'io_options' must be a mapping"):
            DRYESIndex.from_options({}, "bad", {})

    def test_from_options_rejects_invalid_run_options_type(self):
        with pytest.raises(TypeError, match="'run_options' must be a mapping or None"):
            DRYESIndex.from_options({}, {}, "bad")

    def test_from_options_allows_run_options_none(self, monkeypatch):
        monkeypatch.setattr(DRYESIndex, "get_subclass", classmethod(lambda cls, name: _FakeIndex))

        idx = DRYESIndex.from_options(
            {"index_name": "spi"},
            {"data": object(), "index": object()},
            None,
        )

        assert isinstance(idx, _FakeIndex)
        assert idx.run_options == {}

    def test_from_options_requires_index_name_on_base_class(self):
        with pytest.raises(ValueError, match="Missing required 'index_name' or 'index' in index options"):
            DRYESIndex.from_options({}, {"data": object(), "index": object()}, {})
