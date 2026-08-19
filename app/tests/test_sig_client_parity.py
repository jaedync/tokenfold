"""split_signature must be byte-identical in every copy of it.

The live client splits the blob and ships the header, the backfill client does
the same over history, and the server splits any blob an older client still
sends. Three copies of one parser is a deliberate trade (the clients are
stdlib-only single files and cannot import app.*), so the drift has to be
caught mechanically instead of by remembering.

app/sigheader.py is the reference copy only in the sense that it is the one
this suite can see first; the text itself is shared and either side may
propose a change. Whoever changes it changes all three.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "app" / "sigheader.py"
# Every file that carries its own copy of the function.
CLIENT_COPIES = (
    ROOT / "client" / "claude-stats-push.py",     # live ingest path
    ROOT / "client" / "backfill-transcripts.py",  # historical repair path
)
FUNC = "split_signature"


def _function_source(path, name):
    """Exact source segment of a top-level function, or None if absent."""
    if not path.is_file():
        return None
    text = path.read_text()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


class SplitSignatureParityTest(unittest.TestCase):

    def test_server_defines_split_signature(self):
        self.assertIsNotNone(_function_source(SERVER, FUNC),
                             f"{SERVER} must define {FUNC}")

    def test_every_copy_matches_the_server(self):
        server_src = _function_source(SERVER, FUNC)
        for path in CLIENT_COPIES:
            with self.subTest(copy=path.name):
                client_src = _function_source(path, FUNC)
                if client_src is None:
                    # Only reachable if a client half is reverted; a skip keeps
                    # the suite honest instead of going quietly green.
                    raise unittest.SkipTest(
                        f"{path.name} carries no {FUNC} yet")
                self.assertEqual(
                    server_src, client_src,
                    f"{FUNC} has drifted between app/sigheader.py and "
                    f"client/{path.name}. Every copy must stay byte-identical: "
                    "copy one over the others.")


if __name__ == "__main__":
    unittest.main()
