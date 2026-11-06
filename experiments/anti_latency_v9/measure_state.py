from prefix_test import *
def size(x):
 if torch.is_tensor(x):return x.numel()*x.element_size()
 if x is None:return 0
 return sum(size(y) for y in x)
def storage_size(state):
 seen={}
 def visit(x):
  if torch.is_tensor(x):
   storage=x.untyped_storage();seen[(str(x.device),storage.data_ptr())]=storage.nbytes()
  elif x is not None:
   for child in x:visit(child)
 visit(state);return sum(seen.values())
def main():
 data,_=prepare();rows=[]
 with torch.inference_mode():
  for kind in ['attention','signed']:
   for arm in ['long','mixed']:
    m,_=load(kind,2,arm,True)
    for condition,a in data.items():
     _,state=m(a[:4,:-1],cache=True);rows.append({'kind':kind,'arm':arm,'condition':condition,'batch':4,'prefix_tokens':a.shape[1]-1,'persistent_cache_tensor_bytes':size(state),'distinct_retained_cache_storage_bytes':storage_size(state),'model_parameter_bytes':sum(p.numel()*p.element_size() for p in m.parameters()),'warm_position_table_bytes':sum(b.mix._cos.numel()*b.mix._cos.element_size()+b.mix._sin.numel()*b.mix._sin.element_size() for b in m.blocks)})
 (P/'state_memory.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
