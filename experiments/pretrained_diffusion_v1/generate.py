"""Fixed illustrative prompts; not a benchmark or checkpoint selection."""
import run
from run import *
from validate import restore
from tokenizers import Tokenizer
@torch.inference_mode()
def main():
    tok=Tokenizer.from_file(str(P/'original'/'tokenizer.json'));rows=[]
    prompts=['The purpose of scientific research is','Once upon a time, a small robot']
    for kind in ['untouched','normal_finetune','pretrained_diffusion']:
        m=load_original() if kind=='untouched' else restore(kind,1,'equal_cpu' if kind=='pretrained_diffusion' else 'equal_data')
        m.eval()
        for prompt in prompts:
            ids=tok.encode(prompt,add_special_tokens=False).ids;out=[];g=torch.Generator().manual_seed(18286)
            for _ in range(16):
                x=torch.tensor([ids[-32:]])
                if kind=='pretrained_diffusion':
                    logps=[]
                    for _ in range(4):
                        z=torch.randn(1,x.shape[1],128,generator=g)
                        logps.append(m.predict(x,z)[0,-1].log_softmax(-1))
                    logits=torch.logsumexp(torch.stack(logps),0)-math.log(4)
                else:logits=m(x,use_cache=False).logits[0,-1]
                y=int(logits.argmax());ids.append(y);out.append(y)
                if y==0:break
            rows.append(dict(method=kind,prompt=prompt,continuation=tok.decode(out),generated_token_ids=out))
        del m
    write('samples.json',dict(scope='Two fixed illustrative prompts, max16 greedy tokens; seed1 final CPU-matched conversion and normal equal-data control. No ranking or model selection.',samples=rows));print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
