"""Drag-and-drop section reordering: every major dashboard section is a
draggable unit; order persists per client in localStorage and is logged to
the console after each change (so a preferred order can be promoted to the
new server default). Source-level contract tests, like test_dashboard_template."""

import re
import unittest
from pathlib import Path

HTML = (Path(__file__).resolve().parents[2]
        / "templates" / "dashboard.html").read_text()

UNITS = ("cost", "limits", "stats", "sessions", "activity", "trends",
         "models", "machines", "tools", "daily", "reference")


class SectionWrapperTest(unittest.TestCase):
    def test_every_unit_wrapped_once(self):
        # count wrapper tags, not bare attribute mentions (CSS selectors may
        # legitimately reference data-section values)
        for unit in UNITS:
            self.assertEqual(
                HTML.count(f'<div class="tf-section" data-section="{unit}">'),
                1, unit)

    def test_wrapper_count_matches_units(self):
        self.assertEqual(HTML.count('class="tf-section"'), len(UNITS))

    def test_wrappers_open_and_close_balanced(self):
        # crude but effective: the section-wrap region must not change the
        # total div balance of the page
        self.assertEqual(HTML.count("<div"), HTML.count("</div>"))


class ReorderBehaviorTest(unittest.TestCase):
    def test_order_persisted_to_local_storage(self):
        self.assertIn("tf_section_order", HTML)

    def test_order_logged_to_console_after_change(self):
        self.assertIn("console.log", HTML)
        self.assertRegex(HTML, r"console\.log\([^)]*section order")

    def test_saved_order_validated_against_known_sections(self):
        # unknown ids filtered out, missing ids appended — a stale saved
        # order from an older template version must never lose sections
        self.assertIn("applySectionOrder", HTML)
        self.assertRegex(HTML, r"filter\(")

    def test_drag_wiring_present(self):
        for needle in ("dragstart", "dragover", "drop", "setDragImage",
                       "tf-drag"):
            self.assertIn(needle, HTML, needle)

    def test_keyboard_reorder_supported(self):
        self.assertIn("ArrowUp", HTML)
        self.assertIn("ArrowDown", HTML)

    def test_jump_nav_resyncs_to_order(self):
        self.assertIn("syncJumpNav", HTML)

    def test_reduced_motion_respected(self):
        self.assertIn("prefers-reduced-motion", HTML)
        renderer = re.search(r"function applySectionOrder[\s\S]{0,6000}", HTML)
        self.assertIsNotNone(renderer)

    def test_handles_hidden_on_touch(self):
        # native HTML5 DnD does not fire on touch; coarse pointers read the
        # saved order but don't get a broken affordance
        self.assertIn("pointer:coarse", HTML.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
