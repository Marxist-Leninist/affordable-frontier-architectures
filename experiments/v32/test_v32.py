import unittest
from run import *

class MixtureTests(unittest.TestCase):
    def test_complete_distribution_causality(self):
        raw = b'abracadabra abracadabra abracadabra'
        cache = CompactCache()
        for pos, byte in enumerate(raw):
            q,n,k = cache.distribution()
            # Query before feeding current byte: independent brute force over seen history.
            observed = raw[:pos]
            expected = np.full(256,1/256)
            expected_count = 0
            expected_order = 0
            for order in (16,8,4):
                if len(observed) < order:
                    continue
                context = observed[-order:]
                followers = [observed[j+order] for j in range(len(observed)-order) if observed[j:j+order] == context]
                if followers:
                    for follower in followers:
                        expected[follower] += 1
                    expected_count = len(followers)
                    expected_order = order
                    break
            expected /= expected_count+1
            np.testing.assert_array_equal(q,expected)
            self.assertEqual((n,k),(expected_count,expected_order))
            cache.feed(byte)
    def test_gate_never_uses_target(self):
        selection = {'local':dict(constant=.3,by_order={'4':.1,'8':.5,'16':.9})}
        np.testing.assert_array_equal(weights('by_order','local',np.array([0,4,8,16]),selection),[0,.1,.5,.9])
    def test_normalization_and_fallback(self):
        neural = np.arange(1,257,dtype=float); neural /= neural.sum()
        copy = neural[::-1]
        for weight in GRID:
            mixed = (1-weight)*neural+weight*copy
            self.assertAlmostEqual(mixed.sum(),1.)
            np.testing.assert_allclose(np.exp2(-mixture_loss(-np.log2(neural),copy,weight)),mixed,rtol=1e-14)
        np.testing.assert_allclose(mixture_loss(-np.log2(neural),copy,0),-np.log2(neural),rtol=1e-14)
    def test_calibration_separable_objective(self):
        # Exhaustive small grid confirms per-bucket optimum equals joint optimum.
        neural = np.array([.2,.5,.1,.8]); copy=np.array([.8,.2,.3,.6]); order=np.array([4,4,8,8]); grid=[0.,.5,1.]
        chosen=[min(grid,key=lambda w: -np.log2((1-w)*neural[order==k]+w*copy[order==k]).sum()) for k in (4,8)]
        best=min((-np.log2((1-np.where(order==4,a,b))*neural+np.where(order==4,a,b)*copy).sum(),a,b) for a in grid for b in grid)
        self.assertEqual(chosen,list(best[1:]))
if __name__ == '__main__':
    unittest.main()
