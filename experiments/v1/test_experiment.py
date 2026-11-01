import unittest
from experiment import Economy,Architecture
class Tests(unittest.TestCase):
    def test_frontier_exact(self):
        for m in (0,.3,.5):
            e=Economy(margin=m); self.assertAlmostEqual(e.cost(e.required_decode()),e.price/(1+e.tax)*(1-e.margin))
    def test_prefill_alone_can_fail(self):self.assertIsNone(Economy(margin=.9).required_decode())
    def test_cache_preserves_output(self):self.assertLess(Economy(cache_hit=.5).required_decode(),Economy().required_decode())
    def test_idle_hurts(self):self.assertGreater(Economy(utilization=.3).required_decode(),Economy().required_decode())
    def test_state_constant(self):
        a=Architecture('ssm',150e9,0,1);self.assertEqual(a.sequence_bytes(8192),a.sequence_bytes(131072))
    def test_attention_grows(self):
        a=Architecture('attn',150e9,1);self.assertEqual(a.sequence_bytes(131072),16*a.sequence_bytes(8192))
    def test_precision_fails_memory(self):self.assertFalse(Architecture('fp8',150e9,0,1,bits=8).gate(8192,16)['memory_gate'])
    def test_no_unearned_sparse_batch_reuse(self):
        a=Architecture('moe',150e9,1);self.assertGreater(a.gate(8192,64,overlap_exponent=1)['step_time_lower_bound_ms'],a.gate(8192,64,overlap_exponent=0)['step_time_lower_bound_ms'])
    def test_dense_compute_bound(self):self.assertFalse(Architecture('dense',3e12,0,1).gate(8192,128)['20tps_latency_gate'])
    def test_more_state_less_batch(self):self.assertLess(Architecture('big',150e9,0,4).gate(8192,64)['memory_capacity_sequences'],Architecture('small',150e9,0,1).gate(8192,64)['memory_capacity_sequences'])
    def test_invalid(self):
        with self.assertRaises(ValueError):Economy(utilization=0)
        with self.assertRaises(ValueError):Architecture('bad',4e12,1)
if __name__=='__main__':unittest.main()
