from experiment import *
from accounting import restore

def main():
    ds=data();rows=[]
    for name in ['untouched','normal_101_equal_data']:
        t=time.process_time();m=Treatment('normal') if name=='untouched' else restore('normal',101,'equal_data');load=time.process_time()-t
        r=evaluate(m,ds['test'],'test');old=json.loads((P/f'{name}.json').read_text())['test'];difference=max(abs(a-b) for a,b in zip(r['window_nll_nats'],old['window_nll_nats']));assert difference==0.,difference
        rows.append({'name':name,'test_nll':r['nll_nats'],'max_window_difference':difference,'load_cpu_seconds':load,'eval_cpu_seconds':r['cpu_seconds']});print(rows[-1],flush=True);del m
    write('startup_parity.json',{'passed':True,'records':rows,'change':'Explicit NumPy-before-Torch startup through launch.py, matching v2 import order. Scientific implementation files unchanged.'})
if __name__=='__main__':main()
