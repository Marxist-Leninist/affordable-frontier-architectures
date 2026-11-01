"""Functional recurrent-state probe ONLY: untrained recurrence, no LM benchmark."""
from array import array
from time import perf_counter
import json, math
from pathlib import Path

def advance(state,tokens):
    for token in tokens:
        for j in range(len(state)):
            state[j]=.97*state[j]+.03*math.sin(token*(j+1))
    return state

def run():
    cases=[]
    for length in (512,8192):
        tokens=[(i%97)/97 for i in range(length)]
        start=perf_counter(); whole=advance(array('d',[0]*64),tokens);elapsed=perf_counter()-start
        chunked=array('d',[0]*64)
        for i in range(0,length,127):advance(chunked,tokens[i:i+127])
        assert whole==chunked,'streaming/chunk equivalence failed'
        snapshot=array('d',whole)
        advance(whole,[.4,.5]);assert snapshot!=whole,'branch isolation failed'
        cases.append({'sequence_length':length,'state_payload_bytes':snapshot.itemsize*len(snapshot),'cpu_seconds':elapsed,'chunk_equivalent':True,'branch_snapshot_independent':True})
    report={'warning':'Toy untrained diagonal recurrence, not Mamba, not LM quality or GPU performance; token-list memory excluded from state payload','cases':cases}
    path=Path(__file__).parent/'outputs'/'cpu_state_probe.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':run()
