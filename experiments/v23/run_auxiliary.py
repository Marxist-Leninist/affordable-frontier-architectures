import subprocess,resource,time,json,sys
from pathlib import Path
p=Path(__file__).parent;out=[]
for filename in ['summarize.py','audit_results.py','write_report.py']:
 before=resource.getrusage(resource.RUSAGE_CHILDREN);t=time.perf_counter()
 with (p/(filename+'.log')).open('w') as f:code=subprocess.call([sys.executable,str(p/filename)],stdout=f,stderr=subprocess.STDOUT)
 after=resource.getrusage(resource.RUSAGE_CHILDREN);out.append(dict(script=filename,returncode=code,wall_seconds=time.perf_counter()-t,user_seconds=after.ru_utime-before.ru_utime,system_seconds=after.ru_stime-before.ru_stime));assert code==0
(p/'auxiliary_runtime.json').write_text(json.dumps(out,indent=2)+'\n')
