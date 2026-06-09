"""Security response headers — defense-in-depth backstop (Task F4).

Verifies that every response carries the three cheap headers:
  X-Content-Type-Options: nosniff
  X-Frame-Options: DENY
  Referrer-Policy: same-origin

No CSP: the inline-script-heavy dashboard would require 'unsafe-inline',
defeating it; the actual XSS hole is fixed via </script>-encoding + esc()
in dashboard.py/template.
"""
import unittest

from app.tests._support import TempDBTestCase


class SecurityHeadersTest(TempDBTestCase):
    """Assert all three headers are present on both an API route and the root page."""

    def test_security_headers_on_api_route(self):
        """GET /health must carry all three security headers."""
        c = self.client()
        r = c.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.headers.get("x-content-type-options"),
            "nosniff",
            "X-Content-Type-Options missing or wrong on /health",
        )
        self.assertEqual(
            r.headers.get("x-frame-options"),
            "DENY",
            "X-Frame-Options missing or wrong on /health",
        )
        self.assertEqual(
            r.headers.get("referrer-policy"),
            "same-origin",
            "Referrer-Policy missing or wrong on /health",
        )

    def test_security_headers_on_root(self):
        """GET / (dashboard) must carry all three security headers."""
        c = self.client()
        r = c.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(
            r.headers.get("x-content-type-options"),
            "nosniff",
            "X-Content-Type-Options missing or wrong on /",
        )
        self.assertEqual(
            r.headers.get("x-frame-options"),
            "DENY",
            "X-Frame-Options missing or wrong on /",
        )
        self.assertEqual(
            r.headers.get("referrer-policy"),
            "same-origin",
            "Referrer-Policy missing or wrong on /",
        )


if __name__ == "__main__":
    unittest.main()
