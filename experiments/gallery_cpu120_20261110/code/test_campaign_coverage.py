"""Coverage regressions: variants/seeds are not distinct source architectures."""
import copy
import unittest
from campaign_coverage import classify_run, UPSTREAM_BY_PLUGIN, SIM_PREFIX

class CoverageTests(unittest.TestCase):
    def fixture(self):
        return {"plugin":"fusion879","seed":1,"ok":True,"steps":1200,"objective":"agillm-fit-v1",
                "sim_sha256":SIM_PREFIX,"causality":{"ok":True},"skipped":0,
                "cfg":{"V":8192,"d":256,"H":4,"L":12,"ctx":256,"batch":8},
                "curve":[{"step":1200,"tokens":2457600,"ar":5.0,"sat":5.2,"nat":5.8}]}
    def test_eligible_fixed_endpoint(self):
        self.assertEqual(classify_run(self.fixture()),("eligible_agillm_simulator_run",[]))
    def test_operator_test_not_training(self):
        r=self.fixture();r.update(plugin="upstream_tests",steps=0)
        self.assertEqual(classify_run(r)[0],"operator_check_only")
    def test_variants_share_upstream_identity(self):
        self.assertEqual({UPSTREAM_BY_PLUGIN[x] for x in ("fusion879","fusion879_lr","fusion879_lrk","fusion879_hn")},{879})
        self.assertNotIn("base",UPSTREAM_BY_PLUGIN)
        self.assertNotIn("nope",UPSTREAM_BY_PLUGIN)
    def test_incomplete_excluded(self):
        r=self.fixture();r["steps"]=150
        self.assertNotEqual(classify_run(r)[0],"eligible_agillm_simulator_run")
    def test_failed_or_skipped_excluded(self):
        for key,value in (("ok",False),("skipped",1)):
            r=self.fixture();r[key]=value
            self.assertNotEqual(classify_run(r)[0],"eligible_agillm_simulator_run")
    def test_nonfinite_metrics_excluded(self):
        for value in (float("nan"),float("inf"),None,True):
            r=self.fixture();r["curve"][0]["sat"]=value
            self.assertNotEqual(classify_run(r)[0],"eligible_agillm_simulator_run")
    def test_wrong_protocol_excluded(self):
        for key,value in (("objective","different"),("sim_sha256","bad")):
            r=self.fixture();r[key]=value
            self.assertNotEqual(classify_run(r)[0],"eligible_agillm_simulator_run")
        r=self.fixture();r["cfg"]["V"]=256
        self.assertNotEqual(classify_run(r)[0],"eligible_agillm_simulator_run")
    def test_missing_endpoint_excluded(self):
        r=self.fixture();r["curve"]=[]
        self.assertNotEqual(classify_run(r)[0],"eligible_agillm_simulator_run")
    def test_metadata_not_training(self):
        self.assertEqual(classify_run({"summary":{"total":105}})[0],"non_run")

if __name__=="__main__":
    unittest.main()
