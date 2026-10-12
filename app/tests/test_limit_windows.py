"""The shared limit-window helpers live in a module with no routes, and every
importer uses that one copy (no route module imports another for them)."""

import unittest

import app.api
import app.ha
import app.limit_readings
import app.limit_windows as lw
import app.spend_history


class LimitWindowsModuleTest(unittest.TestCase):
    def test_module_has_no_routes(self):
        self.assertFalse(hasattr(lw, "router"))

    def test_importers_share_one_copy(self):
        for mod in (app.api, app.ha):
            self.assertIs(mod._bucket_window_start, lw._bucket_window_start)
        for mod in (app.api, app.ha, app.limit_readings, app.spend_history):
            self.assertIs(mod._iso_to_epoch, lw._iso_to_epoch)
        for mod in (app.api, app.limit_readings):
            self.assertIs(mod._scrub_to_minute_or_none,
                          lw._scrub_to_minute_or_none)

    def test_limit_readings_imports_no_route_module_helpers(self):
        self.assertEqual(
            app.limit_readings._scrub_to_minute_or_none.__module__,
            "app.limit_windows")


if __name__ == "__main__":
    unittest.main()
