"""MCP tool display naming (UX P2-15).

Raw MCP tool ids like ``mcp__plugin_playwright_playwright__browser_take_screenshot``
consumed half the Tool Usage chart width. The aggregator now display-names them
as ``server · tool`` (mcp__ prefix stripped, repeated server tokens deduped),
keeps the full ids in ``tool_full_names`` for tooltips, and merges counts when
several raw ids collapse to one display name.
"""

import unittest
from collections import Counter

from app.aggregator import _display_tool_counts, display_tool_name


class DisplayToolNameTest(unittest.TestCase):

    def test_plain_tools_untouched(self):
        for name in ("Bash", "Read", "Edit", "WebFetch"):
            self.assertEqual(display_tool_name(name), name)

    def test_mcp_tool_server_dot_tool(self):
        self.assertEqual(
            display_tool_name("mcp__plugin_playwright_playwright__browser_take_screenshot"),
            "playwright · browser_take_screenshot")

    def test_repeated_server_tokens_deduped(self):
        self.assertEqual(
            display_tool_name("mcp__imessage__search_messages"),
            "imessage · search_messages")
        self.assertEqual(
            display_tool_name("mcp__actual_budget__get_transactions"),
            "actual_budget · get_transactions")

    def test_mcp_prefix_only(self):
        self.assertEqual(display_tool_name("mcp__solo"), "solo")

    def test_empty_and_none_safe(self):
        self.assertIsNone(display_tool_name(None))
        self.assertEqual(display_tool_name(""), "")


class DisplayToolCountsTest(unittest.TestCase):

    def test_merges_counts_and_keeps_full_names(self):
        raw = Counter({
            "Bash": 100,
            "mcp__plugin_playwright_playwright__browser_click": 30,
            "mcp__plugin_playwright_playwright__browser_navigate": 20,
        })
        counts, fulls = _display_tool_counts(raw)
        self.assertEqual(counts["Bash"], 100)
        self.assertEqual(counts["playwright · browser_click"], 30)
        self.assertEqual(
            fulls["playwright · browser_click"],
            ["mcp__plugin_playwright_playwright__browser_click"])
        self.assertNotIn("Bash", fulls)  # only renamed tools get a full-name entry

    def test_top_n_applied_after_merge(self):
        raw = Counter({f"tool{i}": i for i in range(1, 30)})
        counts, _ = _display_tool_counts(raw, top=5)
        self.assertEqual(len(counts), 5)
        self.assertEqual(list(counts)[0], "tool29")


if __name__ == "__main__":
    unittest.main()
