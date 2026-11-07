"""Validation-only decoder ablations and illustrative fixed-prompt generations."""
import run
from run import *
from accounting import restore
from tokenizers import Tokenizer
@torch.inference_mode()
def main():
    d=data();rows=[]
    for kind in ['direct_ce','direct_kd']:
        model=restore(kind,11,'equal_data');model.eval()
        for decoder in ['high_noise','raw_rollout']:
            r=evaluate(model,kind,d['validation'],'validation',decoder,4)
            rows.append(dict(kind=kind,endpoint='equal_data',seed=11,**r));print(kind,decoder,r['nll_nats'],flush=True)
        del model
    write('diagnostic.json',dict(scope='Prespecified seed-11 equal-data decoder diagnostics on validation only. No checkpoint/decoder selection, no replacement of primary test scores.',results=rows))
    tokenizer=Tokenizer.from_file(str(P/'original'/'tokenizer.json'));samples=[]
    for kind in ['untouched','normal_ce','normal_kd','direct_ce','direct_kd']:
        model=load_original() if kind=='untouched' else restore(kind,11,'equal_data');model.eval()
        for prompt in ['The purpose of scientific research is','Once upon a time, a small robot']:
            ids=tokenizer.encode(prompt,add_special_tokens=False).ids;out=[];g=torch.Generator().manual_seed(811014)
            for _ in range(12):
                x=torch.tensor([ids[-32:]])
                if kind.startswith('direct'):
                    ctx=model.prepare_context(x);logps=[]
                    for _ in range(4):
                        noise=torch.randn(1,x.shape[1],D,generator=g)
                        logps.append(model.predict(x,noise,'normalized_rollout',ctx)[0,-1].log_softmax(-1))
                    logits=torch.logsumexp(torch.stack(logps),0)-math.log(4)
                else:logits=model(x,use_cache=False).logits[0,-1]
                token=int(logits.argmax());ids.append(token);out.append(token)
                if token==0:break
            samples.append(dict(kind=kind,seed=11,prompt=prompt,continuation=tokenizer.decode(out),token_ids=out))
        del model
    write('samples.json',dict(scope='Two fixed illustrative prompts, greedy12tokens,seed11equal-data models; not a generation-quality benchmark or selection criterion.',samples=samples))
if __name__=='__main__':main()
