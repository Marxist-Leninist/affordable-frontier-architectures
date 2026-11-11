"""Static, partial forward MAC count. This does not execute Torch."""
import json
from pathlib import Path

def count(batch=8,t=64,d=48,ff=96,depth=4,vocab=256):
    attention_dense=4*t*d*d
    attention_scores_values=2*t*t*d
    mlp=2*t*d*ff
    head=t*d*vocab
    per_sample=depth*(attention_dense+attention_scores_values+mlp)+head
    param=256*d+t*d+depth*((2*d)+(3*d*d+3*d)+(d*d+d)+(2*d)+(d*ff+ff)+(ff*d+d))+2*d
    return {'shape':{'batch':batch,'context':t,'width':d,'ffn':ff,'depth':depth,'vocabulary':vocab},'parameter_count':param,'per_sample_forward_matrix_macs':per_sample,'per_batch_forward_matrix_macs':batch*per_sample,'per_block_per_sample':{'attention_dense_macs':attention_dense,'full_square_attention_score_value_macs':attention_scores_values,'mlp_macs':mlp},'head_per_sample_macs':head,'scope':'Identical module matrix shapes across ordinary, reversible graph and reversible inverse arms. Dense square attention estimate, not actual fused kernel operations.','excluded':['bias additions','normalization','GELU','softmax','residual additions','stream averages','embedding lookup','backward','optimizer','memory movement','dispatch'],'caveats':['One MAC means multiply-accumulate, not a full FLOP count.','Causal attention implementations can elide masked work; this is a static full-square convention.','Inverse backprop recomputes attention and MLP forward operations, so equal forward MAC count does not mean equal training compute.','There is no second stored feed-forward graph for inverse mode, but recomputation produces temporary local graphs.']}
if __name__=='__main__':
    x=count();Path(__file__).with_name('compute_accounting.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
