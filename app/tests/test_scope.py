"""Tests for app.config scope partitioning: scope_predicate, resolve_scope,
ENTERPRISE_PRED / PERSONAL_PRED as a TRUE partition, and env lock."""

import unittest
from unittest.mock import patch


class ScopePredicateTest(unittest.TestCase):
    """scope_predicate returns correct SQL strings; invalid scope raises InvalidScope."""

    def test_enterprise_returns_enterprise_pred(self):
        from app.config import scope_predicate, ENTERPRISE_PRED
        self.assertEqual(scope_predicate("enterprise"), ENTERPRISE_PRED)

    def test_personal_returns_personal_pred(self):
        from app.config import scope_predicate, PERSONAL_PRED
        self.assertEqual(scope_predicate("personal"), PERSONAL_PRED)

    def test_bogus_scope_raises_invalid_scope(self):
        from app.config import scope_predicate, InvalidScope
        with self.assertRaises(InvalidScope):
            scope_predicate("bogus")

    def test_empty_scope_raises_invalid_scope(self):
        from app.config import scope_predicate, InvalidScope
        with self.assertRaises(InvalidScope):
            scope_predicate("")


class PartitionPredicateTest(unittest.TestCase):
    """ENTERPRISE_PRED and PERSONAL_PRED are exact complements and use COALESCE."""

    def test_enterprise_pred_uses_coalesce(self):
        from app.config import ENTERPRISE_PRED
        self.assertIn("COALESCE", ENTERPRISE_PRED)

    def test_personal_pred_is_complement(self):
        from app.config import ENTERPRISE_PRED, PERSONAL_PRED
        # PERSONAL_PRED must be NOT(...ENTERPRISE_PRED...)
        self.assertIn("NOT (", PERSONAL_PRED)
        self.assertIn(ENTERPRISE_PRED, PERSONAL_PRED)


class ResolveScopeTest(unittest.TestCase):
    """resolve_scope: returns effective scope, respects LOCKED_SCOPE, raises on bad input."""

    def test_no_lock_none_request_returns_default(self):
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", None):
            from app.config import resolve_scope
            self.assertEqual(resolve_scope(None), "enterprise")

    def test_no_lock_personal_request_returns_personal(self):
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", None):
            from app.config import resolve_scope
            self.assertEqual(resolve_scope("personal"), "personal")

    def test_no_lock_enterprise_request_returns_enterprise(self):
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", None):
            from app.config import resolve_scope
            self.assertEqual(resolve_scope("enterprise"), "enterprise")

    def test_no_lock_bogus_request_raises_invalid_scope(self):
        import app.config as cfg
        from app.config import InvalidScope
        with patch.object(cfg, "LOCKED_SCOPE", None):
            from app.config import resolve_scope
            with self.assertRaises(InvalidScope):
                resolve_scope("bogus")

    def test_locked_enterprise_none_request_returns_enterprise(self):
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            from app.config import resolve_scope
            self.assertEqual(resolve_scope(None), "enterprise")

    def test_locked_enterprise_enterprise_request_returns_enterprise(self):
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            from app.config import resolve_scope
            self.assertEqual(resolve_scope("enterprise"), "enterprise")

    def test_locked_enterprise_personal_request_raises_scope_locked(self):
        import app.config as cfg
        from app.config import ScopeLocked
        with patch.object(cfg, "LOCKED_SCOPE", "enterprise"):
            from app.config import resolve_scope
            with self.assertRaises(ScopeLocked):
                resolve_scope("personal")

    def test_locked_personal_none_request_returns_personal(self):
        import app.config as cfg
        with patch.object(cfg, "LOCKED_SCOPE", "personal"):
            from app.config import resolve_scope
            self.assertEqual(resolve_scope(None), "personal")

    def test_locked_personal_enterprise_request_raises_scope_locked(self):
        import app.config as cfg
        from app.config import ScopeLocked
        with patch.object(cfg, "LOCKED_SCOPE", "personal"):
            from app.config import resolve_scope
            with self.assertRaises(ScopeLocked):
                resolve_scope("enterprise")

    def test_invalid_locked_scope_raises_invalid_scope(self):
        """A bad LOCKED_SCOPE env value should raise InvalidScope, not silently return it."""
        import app.config as cfg
        from app.config import InvalidScope
        with patch.object(cfg, "LOCKED_SCOPE", "badvalue"):
            from app.config import resolve_scope
            with self.assertRaises(InvalidScope):
                resolve_scope(None)


if __name__ == "__main__":
    unittest.main()
