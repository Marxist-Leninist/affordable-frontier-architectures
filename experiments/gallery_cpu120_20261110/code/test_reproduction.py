"""Fresh-checkout/report-schema regression tests, with no model execution or network."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
E = HERE.parent

class ReportRegressionTests(unittest.TestCase):
    def generate(self, report):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            verify = td / "fresh_verify.json"
            verify.write_text(json.dumps(report))
            out = td / "README.md"
            subprocess.run([sys.executable, str(HERE / "make_readme.py"),
                str(E / "METRICS.json"), str(verify), str(E / "SMOKE.json"),
                str(E / "STATIC_AUDIT.json"), str(HERE / "README_static.md"), str(out)],
                check=True, capture_output=True)
            return out.read_text()

    def test_static_audit_creates_missing_parents_without_executing(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            source = td / "gallery.json"
            source.write_text(json.dumps([{"index": 1, "name": "test",
                "program": "class DeltaNet:\n    pass\n"}]))
            out = td / "new" / "work" / "STATIC_AUDIT.json"
            subprocess.run([sys.executable, str(HERE / "static_audit.py"),
                str(source), str(out)], check=True, capture_output=True)
            self.assertEqual(json.loads(out.read_text())["clean"], 1)

    def test_minimal_current_verifier_schema_does_not_invent_census(self):
        report = {"checked": 6, "reproduce": 6, "do_not_reproduce": [],
                  "tolerance_nats": 1e-6, "rows": []}
        text = self.generate(report)
        self.assertIn("6 passed its recorded tolerance", text)
        self.assertIn("does not contain the optional gallery-wide counter census", text)
        self.assertIn("No reconstruction count is inferred", text)
        self.assertIn("140 from the original cohort", text)
        self.assertNotIn("match the recorded values exactly", text)

    def test_failed_verification_is_reported_without_exact_match_claim(self):
        report = {"checked": 6, "reproduce": 5, "do_not_reproduce": ["bad"],
                  "tolerance_nats": 1e-6, "rows": [],
                  "largest_abs_diff_nats": 0.00042}
        text = self.generate(report)
        self.assertIn("1 did not reproduce", text)
        self.assertIn("0.00042 nats", text)
        self.assertNotIn("difference 0.0", text)

    def test_legacy_combined_provenance_remains_explicit(self):
        text = self.generate(json.loads((E / "VERIFY.json").read_text()))
        self.assertIn("33 of the 105 programs", text)
        self.assertIn("9 checkpoints whose runtime counters were reconstructed", text)
        self.assertIn("historical provenance", text)
        self.assertIn("146 freshly rebuilt models", text)

if __name__ == "__main__":
    unittest.main()
