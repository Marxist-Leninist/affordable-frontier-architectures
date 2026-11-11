"""Bounded correctness audit; never benchmarks or changes trained model code."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import copy, hashlib, inspect, json, unittest
from pathlib import Path
import torch
from models import NAT, MambaDirection, MambaBlock, AttentionBlock, MASK, count, scan_parallel, scan_sequential

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parent
DETAIL={}

class Audit(unittest.TestCase):
    def test_scan_outputs_and_gradients(self):
        torch.manual_seed(42)
        errors=[]
        for length in [1,2,3,7,16,48]:
            a=(torch.rand(2,length,3,4,dtype=torch.float64)*.99).requires_grad_()
            b=torch.randn_like(a,requires_grad=True)
            weight=torch.randn_like(a)
            pa=scan_parallel(a,b); se=scan_sequential(a,b)
            gp=torch.autograd.grad((pa*weight).sum(),(a,b),retain_graph=True,allow_unused=True,materialize_grads=True)
            gs=torch.autograd.grad((se*weight).sum(),(a,b),allow_unused=True,materialize_grads=True)
            torch.testing.assert_close(pa,se,rtol=1e-11,atol=1e-11)
            for x,y in zip(gp,gs): torch.testing.assert_close(x,y,rtol=1e-11,atol=1e-11)
            errors.append({'length':length,'output_max_abs':(pa-se).abs().max().item(),'gradient_max_abs':max((x-y).abs().max().item() for x,y in zip(gp,gs))})
        DETAIL['scan_float64']=errors

    def test_direction_matches_sequential_backward(self):
        torch.manual_seed(43)
        a=MambaDirection(d=8,state=3).double(); b=copy.deepcopy(a); b.scan=scan_sequential
        x=torch.randn(2,7,8,dtype=torch.float64,requires_grad=True); y=x.detach().clone().requires_grad_()
        oa=a(x); ob=b(y); torch.testing.assert_close(oa,ob,rtol=1e-10,atol=1e-12)
        oa.square().sum().backward();ob.square().sum().backward()
        torch.testing.assert_close(x.grad,y.grad,rtol=1e-9,atol=1e-11)
        for pa,pb in zip(a.parameters(),b.parameters()): torch.testing.assert_close(pa.grad,pb.grad,rtol=1e-9,atol=1e-11)
        DETAIL['mamba_equivalence']='Outputs, input gradients, and all parameter gradients match sequential recurrence.'

    def test_causal_direction_bidirectional_block(self):
        torch.manual_seed(44)
        m=MambaDirection(d=8,state=3).double()
        x=torch.randn(1,7,8,dtype=torch.float64);y=x.clone();y[:,4:]+=1
        torch.testing.assert_close(m(x)[:,:4],m(y)[:,:4],rtol=0,atol=0)
        block=MambaBlock(8).double();xx=torch.randn(1,7,8,dtype=torch.float64,requires_grad=True)
        out=block(xx); z=block.norm(xx)
        explicit=xx+(block.forward_dir(z)+block.reverse_dir(z.flip(1)).flip(1))/(2**.5)
        torch.testing.assert_close(out,explicit)
        out[0,0,0].backward();self.assertGreater(xx.grad[0,-1].abs().sum().item(),0)
        DETAIL['bidirectionality']='Forward direction causal; block equals explicit forward-plus-reversed scan and first position depends on last.'

    def test_nat_interface_branches_parameters(self):
        torch.manual_seed(45)
        self.assertEqual(list(inspect.signature(NAT.forward).parameters),['self','prefix','out_len'])
        sizes={}
        for kind,cls in [('transformer',AttentionBlock),('mamba',MambaBlock)]:
            model=NAT(kind).eval(); self.assertTrue(all(isinstance(b,cls) for b in model.blocks)); sizes[kind]=count(model)
            inputs=[]; calls=[]
            hook=model.embed.register_forward_pre_hook(lambda m,args:inputs.append(args[0].detach().clone()))
            hooks=[b.register_forward_hook(lambda m,a,o:calls.append(type(m).__name__)) for b in model.blocks]
            prefix=torch.randint(0,256,(2,32));before=prefix.clone()
            with torch.no_grad(): output=model(prefix)
            self.assertEqual(output.shape,(2,16,256));self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(len(inputs),1);self.assertEqual(len(calls),2)
            self.assertTrue(torch.equal(inputs[0][:,:32],prefix));self.assertTrue((inputs[0][:,32:]==MASK).all());self.assertTrue(torch.equal(before,prefix))
            hook.remove()
            for h in hooks:h.remove()
        self.assertLess(abs(sizes['transformer']-sizes['mamba'])/sizes['transformer'],.002)
        DETAIL['parameter_counts']=sizes
        DETAIL['target_leakage']='Forward accepts prefix and output length only; observed embedded suffix consists solely of MASK. Each block called once.'

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Audit)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':[str(x) for x in result.failures],'errors':[str(x) for x in result.errors],'details':DETAIL,'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['models.py','run.py']},'scope':'Tiny CPU unit tests, no performance benchmarking, no live service access. Dense FP32 tiny architecture ablation, not routed/quantized AGILLM or 600B validation.','review_findings':['Mamba-1 selective recurrence with input-dependent delta/B/C, diagonal negative A, causal depthwise convolution, D skip, SiLU gate. Bidirectionality is an explicit independent-two-direction extension.','Fixed-step same-seed training uses deterministic identical NumPy data batches; verify stored hashes after fits. LR selection occurs on validation, test data first evaluated after final fits.','5 s snapshots are first whole step crossing budget, not strict identical elapsed times. Compare recorded elapsed and steps; training timing excludes data preparation, validation, checkpoint saving.','Attention uses optimized PyTorch SDPA while Mamba uses unfused O(T log T) work associative scan with repeated allocations. Timing is implementation-specific CPU performance, not asymptotic or CUDA comparison.','PAD embeddings participate in attention/scan for toy prompts. Same inputs for both models, no padding mask; not a leakage channel.','Text exact48 screening covers only actual 2 MB training prefix and within-evaluation duplicates; shorter/natural phrase overlap and same-source correlation remain. Byte suffix loss measures factorized conditional prediction, not autoregressive perplexity or general intelligence.','Three seeds and tiny training budget limit uncertainty claims; test-example resampling alone must not imply training-seed uncertainty.','NAT kind values other than transformer silently select Mamba; current explicit call sites use valid strings. out_len=0 unsupported by slicing; fixed protocol uses 16.']}
    (ROOT/'audit.json').write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(not result.wasSuccessful())
