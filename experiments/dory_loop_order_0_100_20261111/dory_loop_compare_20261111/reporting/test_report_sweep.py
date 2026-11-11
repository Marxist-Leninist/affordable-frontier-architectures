"""Synthetic fixtures only. No tensor imports, real sweep/model/data reads."""
import json,tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from report_sweep import PHASES,KINDS,SEEDS,DOMAINS,generate,collect,dump,read

def fixture(src):
    src.mkdir();ids={f'{k}_{s}':str(s)*64 for k in KINDS for s in SEEDS}
    selection=dict(status='selected',windows_per_domain=16,checkpoint_hashes=ids)
    dump(src/'cohort_selection.json',selection);dump(src/'status.json',{'runs':[dict(kind=k,seed=s) for k in KINDS for s in SEEDS]})
    dump(src/'manifest.json',dict(selection=selection,cohorts={p:{d:dict(windows=16 if p=='all101' else 128,selection_sha256='a'*64,source={'private_text':'NEVER_EXPORT_THIS'}) for d in DOMAINS} for p in PHASES}))
    for kind in KINDS:
        for seed in SEEDS:
            rows=[]
            for phase,counts in PHASES.items():
                for loops in counts:
                    datasets={d:dict(status='completed',evaluated_windows=16 if phase=='all101' else 128,suffix_bpb=3+seed*.1+loops*.001,byte_accuracy=.3,exact_32_byte_match=0.,forward_cpu_seconds=loops*.01,forward_wall_seconds=loops*.02,per_window_loss_nats=['NEVER_EXPORT_THIS'],groups=['NEVER_EXPORT_THIS']) for d in DOMAINS}
                    rows.append(dict(phase=phase,loops=loops,trained_loops=2,unique_parameters=764304,invoked_module_parameters=353424 if loops==0 else 764304,applied_layers=16+20*loops,recurrent_applications=20*loops,datasets=datasets))
            dump(src/f'{kind}_s{seed}.json',dict(kind=kind,seed=seed,checkpoint_sha256=ids[f'{kind}_{seed}'],rows=rows))

class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.src=self.root/'input';fixture(self.src)
    def tearDown(self):self.temp.cleanup()
    def test_complete(self):
        out=self.root/'report';rows,summary=generate(self.src,out)
        self.assertTrue(all(r['applied_layers']==16+20*r['loops'] and r['unique_parameters']==764304 for r in rows));self.assertTrue(all(r['invoked_module_parameters']==353424 for r in rows if r['loops']==0));self.assertEqual(len(rows),1296);self.assertEqual(len(summary),432)
        self.assertEqual({r['loops'] for r in rows if r['phase']=='all101'},set(range(101)))
        self.assertAlmostEqual(summary[0]['suffix_bpb_seed_sd'],.1)
        for f in out.glob('*.svg'):ET.parse(f)
        self.assertEqual(len(list(out.glob('*.svg'))),8)
        self.assertFalse(any('NEVER_EXPORT_THIS' in f.read_text() for f in out.iterdir()))
        self.assertIn('16 windows per domain',(out/'REPORT.md').read_text())
    def test_missing_nonfinite_capped(self):
        p=self.src/'split_s1.json';o=read(p)
        for index,status in enumerate(('nonfinite_logits','capped','missing_resource_cap')):o['rows'][index]['datasets']['test']={'status':status}
        o['rows'][3]['datasets']['test']['suffix_bpb']=-1;del o['rows'][4];dump(p,o)
        (self.src/'whole_s3.json').unlink();rows,summary=generate(self.src,self.root/'report')
        for status in ('nonfinite_logits','capped','missing_resource_cap','invalid_completed_metrics','missing_cell','missing_file'):self.assertTrue(any(r['status']==status for r in rows))
        self.assertTrue(all(r['suffix_bpb'] is None for r in rows if r['status']!='completed'))
        self.assertTrue(any(not r['complete'] for r in summary))
    def test_unfinalized_rejected(self):
        (self.src/'status.json').unlink()
        with self.assertRaises(FileNotFoundError):collect(self.src)
    def test_duplicate_rejected(self):
        p=self.src/'split_s1.json';o=read(p);o['rows'].append(o['rows'][0]);dump(p,o)
        with self.assertRaises(ValueError):collect(self.src)
    def test_wrong_checkpoint_rejected(self):
        p=self.src/'split_s1.json';o=read(p);o['checkpoint_sha256']='b'*64;dump(p,o)
        with self.assertRaises(ValueError):collect(self.src)
    def test_manifest_mismatch(self):
        p=self.src/'manifest.json';o=read(p);o['selection']['windows_per_domain']=8;dump(p,o)
        with self.assertRaises(ValueError):collect(self.src)
    def test_wrong_outer_parameter_provenance(self):
        p=self.src/'split_s1.json';o=read(p);o['rows'][0]['invoked_module_parameters']=24720;dump(p,o)
        with self.assertRaises(ValueError):collect(self.src)
    def test_zero_semantics_in_report(self):
        out=self.root/'report';generate(self.src,out);text=(out/'REPORT.md').read_text()
        self.assertIn('These outer attention blocks retain prefix conditioning.',text)
        self.assertIn('353,424',text);self.assertIn('2,016 total layers',text)
    def test_overwrite_rejected(self):
        out=self.root/'report';generate(self.src,out)
        with self.assertRaises(FileExistsError):generate(self.src,out)
if __name__=='__main__':unittest.main()
