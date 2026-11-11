"""Lightweight artificial JSON fixtures only. No experiment evaluation."""
import copy, importlib.util, json, tempfile
from pathlib import Path
spec=importlib.util.spec_from_file_location('report',Path(__file__).with_name('generate_report_long.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def fixture(root):
    p={'seeds':[1,2,3],'final_steps':16000,'task_steps':2000,'tune_seed':101,'tune_steps':1000,'batch_size':8,'architecture':{'transformer_parameters':765568,'mamba_parameters':765184},'learning_rates':[.0005,.0015,.0045],'limitations':['Artificial fixture, not experimental evidence.']}
    selection={'selected_lrs':dict.fromkeys(g.KINDS,.0005)}
    write(root,'protocol_long.json',p);write(root,'frozen_sources.json',{});write(root,'pipeline_status.json',{'status':'training_complete'});write(root,'selection_long.json',selection)
    metrics=lambda:{'n':2,'bpb':4.,'byte_accuracy':.2,'exact_match':0.,'per_example_loss':[2.,3.]}
    desc=lambda:{'mean':4.,'sample_sd':.1,'values':[3.9,4.,4.1]}
    model={'metrics':{},'conditioning_bpb_difference_shuffled_minus_real':desc(),'time_budget_steps':[9000]*3,'time_budget_actual_cpu_seconds':[900.1]*3}
    for key in ('training_cpu_seconds','training_wall_seconds','end_to_end_text_wall_seconds','task_training_cpu_seconds','peak_rss_kib'):model[key]=desc()
    rows=[]
    for kind in g.KINDS:
        for seed in p['seeds']:
            base=f'final_{kind}_s{seed}_lr0.0005'
            r={'status':'completed','steps':16000,'batch_sha256':'paired','initial_common_hashes':{},'training_cpu_seconds':1000.,'compiler_load_seconds':.001,'validation_curves':[{'step':i,'training_cpu_seconds':i/16,'training_wall_seconds':i/12,'validation':{'bpb':4.-i/100000}} for i in range(2000,16001,2000)]}
            task={'status':'completed','steps':2000,'batch_sha256':'paired','training_cpu_seconds':100.}
            write(root,f'runs/{base}/result.json',r);write(root,f'runs/{base}/task_result.json',task)
            for budget in g.BUDGETS:
                datasets=('copy','reverse','sort','prefixsum_mod10') if budget=='separate_task_adaptation' else ('wikitext','fineweb')
                rows.append({'kind':kind,'seed':seed,'budget':budget,'status':'completed','checkpoint_extra':{'training_cpu_seconds':900.1},'metrics':{d:metrics() for d in datasets}})
                model['metrics'][budget]={d:{key:desc() for key in ('bpb','byte_accuracy','exact_match')} for d in datasets}
    for kind in g.KINDS:
        for lr in p['learning_rates']:
            write(root,f'runs/tune_{kind}_s101_lr{lr}/result.json',{'status':'completed','steps':1000,'training_cpu_seconds':10.})
    e={'rows':rows,'unigram_bpb':{'wikitext':4.5,'fineweb':4.6}}
    write(root,'evaluation_results.json',e)
    s={'models':{k:copy.deepcopy(model) for k in g.KINDS},'pairing_verified':{str(i):{} for i in p['seeds']},'paired_document_bootstrap':{'fixed_data_wikitext':{'conditional_document_95_interval':[-.2,.1],'mamba_minus_transformer_bpb':-.1,'seed_paired_means':[-.1,-.05,-.15],'documents':2}}}
    write(root,'summary_long.json',s)
    t={'host':'Synthetic fixture','scope':'Not a benchmark','peak_process_rss_kib':1}
    for k in g.KINDS:t[k]={'train_step_p50_ms':10.,'train_step_p95_ms':11.,'supervised_byte_targets_per_second':25600.,'infer_batch1':{'p50_ms':2.,'p95_ms':3.},'infer_batch8':{'p50_ms':5.,'p95_ms':6.}}
    write(root,'standalone_timings.json',t)
    manifest={d:{'group_ids':[0,1],'evaluated_windows':2,'documents':2,'excluded_exact128_train_or_duplicate':0,'target32_matches_training_all_windows':0,'selection_sha256':'fixture'} for d in ('wikitext','fineweb')}
    write(root,'evaluation/test_manifest.json',manifest)
    write(root,'compute_accounting_long.json',g.read(Path(__file__).with_name('compute_accounting_long.json')))
    (root/'THIRD_PARTY_NOTICES.md').write_text('Synthetic fixture notice, not provenance evidence.')
    write(root,'fast_scan_results.json',{'benchmarks':[],'compile_seconds':.1,'first_use_seconds':.2})
    return e

def main():
    with tempfile.TemporaryDirectory(prefix='nat_report_fixture_') as tmp:
        root=Path(tmp);e=fixture(root)
        data=g.preflight(root);g.render(root,root/'report',data)
        assert len(list((root/'report').glob('*.png')))==5
        assert (root/'report/REPORT.md').exists()
        resource=g.read(root/'report/resource_accounting.json')
        assert resource['final_text_update_cpu_seconds']==6000.
        assert resource['separate_task_update_cpu_seconds']==600.
        assert resource['tuning_update_cpu_seconds']==60.
        assert abs(resource['total_measured_update_cpu_hours']-1.85)<1e-12
        report=(root/'report/REPORT.md').read_text()
        assert '12.8125%' in report and 'not evidence of cumulative arithmetic' in report
        upgrade={'status':'completed','whole_model_training_speedup':2.,'summary':{'native_linear':{'median_seconds':.5},'torch_parallel_reference':{'median_seconds':1.}},'parity_max_absolute_error':{'logits':1e-7,'gradients':1e-7,'updated_parameters':1e-7}}
        write(root,'scan_upgrade_whole_model.json',upgrade)
        g.render(root,root/'report',g.preflight(root))
        assert '2.000× training-step speedup' in (root/'report/REPORT.md').read_text()
        assert 'scan_upgrade_whole_model.json' in g.read(root/'report/report_input_hashes.json')
        write(root,'scan_upgrade_whole_model.json',{'status':'failed_or_cancelled'})
        g.render(root,root/'report',g.preflight(root))
        assert 'No speedup claim is made from it' in (root/'report/REPORT.md').read_text()
        write(root,'pipeline_status.json',{'status':'final_training'})
        try:g.preflight(root)
        except AssertionError:pass
        else:raise AssertionError('Training gate did not reject')
        write(root,'pipeline_status.json',{'status':'training_complete'})
        e['rows'].append(e['rows'][0]);write(root,'evaluation_results.json',e)
        try:g.preflight(root)
        except AssertionError:pass
        else:raise AssertionError('Duplicate row not rejected')
        e['rows'].pop();e['rows'][0]['metrics']['wikitext']['per_example_loss']=[];write(root,'evaluation_results.json',e)
        try:g.preflight(root)
        except AssertionError:pass
        else:raise AssertionError('Misaligned metric arrays not rejected')
    print('PASS: artificial render (5 PNGs + report), training gate, duplicate rows, array alignment. Temporary fixtures deleted.')
if __name__=='__main__':main()
