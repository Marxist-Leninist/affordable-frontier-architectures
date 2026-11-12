"""v21 = v20 + CHAINMIX512: eleven small additive hunks in serve128.py (every anchor must occur exactly once). Run once in
the v21 dir. A request without chain_mix runs exactly the v20 code path."""
import pathlib
import sys

p = pathlib.Path(__file__).resolve().parent / 'serve128.py'
s = p.read_text()
if 'chainmix512' in s:
    sys.exit('serve128.py already carries CHAINMIX512')
if 'loop512' not in s:
    sys.exit('this patch expects the v20 serve128.py (LOOP512 present)')
H = [
 ("import loop512\n", "import loop512\nimport chainmix512\n"),
 ("REV+='+loop512-v1-20261112'\n", "REV+='+loop512-v1-20261112'\nREV+='+chainmix512-v1-20261112'\n"),
 ("loop512.install(model);M=model", "loop512.install(model);chainmix512.install(model.V);M=model"),
 (" d['loop']=loop512.info(M)  # LOOP512: opt-in looped forward (loop512.py)\n",
  " d['loop']=loop512.info(M)  # LOOP512: opt-in looped forward (loop512.py)\n d['chain_mix']=chainmix512.info()  # CHAINMIX512: opt-in Markov chain mixed into the next-token distribution (chainmix512.py)\n"),
 (" loop_plan(req,score,cp is not None)  # LOOP512: validated before the slot\n return v\n",
  " loop_plan(req,score,cp is not None)  # LOOP512: validated before the slot\n chainmix512.parse(req,score)  # CHAINMIX512: validated before the slot\n return v\n"),
 (" if lp is not None:rcal=None  # LOOP512: the calibration was fitted on unlooped states\n",
  " if lp is not None:rcal=None  # LOOP512: the calibration was fitted on unlooped states\n cm=chainmix512.parse(req)  # CHAINMIX512: None = the model alone\n"),
 ("    raw_src=torch.softmax(logits.float(),dim=-1) if topn else None  # TOPLOGPROBS v2: the model's own distribution, before bans, penalties and temperature\n",
  "    raw_src=torch.softmax(logits.float(),dim=-1) if topn else None  # TOPLOGPROBS v2: the model's own distribution, before bans, penalties and temperature\n    if cm is not None:logits=chainmix512.mix_logits(cm,logits,hist)  # CHAINMIX512: log((1-g) model + g chain); raw_top above stays the model alone\n"),
 (" if lp is not None:stats['loop']=loop512.summary(lp,int(M.L))  # LOOP512\n stats['script_controls']=script.metadata()\n",
  " if lp is not None:stats['loop']=loop512.summary(lp,int(M.L))  # LOOP512\n if cm is not None:stats['chain_mix']=chainmix512.summary(cm);stats['sampling_applied'].update(chain_mix=cm['g'])  # CHAINMIX512\n stats['script_controls']=script.metadata()\n"),
 (" if lp is not None and lp['adaptive']:chunk=1  # LOOP512: the stop rule is per token\n",
  " if lp is not None and lp['adaptive']:chunk=1  # LOOP512: the stop rule is per token\n cm=chainmix512.parse(req,True)  # CHAINMIX512: None = the model alone\n"),
 ("   nll.append(-torch.log_softmax(lg,-1).gather(1,tg[:,None])[:,0]);arg.append(lg.argmax(-1))\n",
  "   lgm=lg if cm is None else chainmix512.mix_rows(cm,lg,x,s)  # CHAINMIX512: the mixture, for the loss and top-1 only\n   nll.append(-torch.log_softmax(lgm,-1).gather(1,tg[:,None])[:,0]);arg.append(lgm.argmax(-1))\n"),
 (" if lp is not None:out['loop']=loop512.summary(lp,int(M.L))  # LOOP512\n",
  " if lp is not None:out['loop']=loop512.summary(lp,int(M.L))  # LOOP512\n if cm is not None:out['chain_mix']=chainmix512.summary(cm)  # CHAINMIX512\n"),
]
for old, new in H:
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)
p.write_text(s)
print('patched serve128.py:', len(H), 'hunks')
