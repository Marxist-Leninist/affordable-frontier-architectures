import copy, unittest, torch
class TestStateIsolation(unittest.TestCase):
 def test_forks_do_not_mutate_source_or_other_arm(self):
  m=torch.nn.Linear(2,1);o=torch.optim.AdamW(m.parameters(),lr=.01);m(torch.ones(1,2)).sum().backward();o.step();source=copy.deepcopy(o.state_dict());frozen=copy.deepcopy(source);forks=[]
  for _ in range(2):
   a=torch.nn.Linear(2,1);p=torch.optim.AdamW(a.parameters(),lr=.01);p.load_state_dict(copy.deepcopy(source));forks.append((a,p))
  a,p=forks[0];a(torch.ones(1,2)).sum().backward();p.step()
  for key,state in source['state'].items():
   for name,value in state.items():
    self.assertTrue(torch.equal(value,frozen['state'][key][name]));self.assertTrue(torch.equal(value,forks[1][1].state_dict()['state'][key][name]))
  self.assertEqual({v['step'].item() for v in p.state_dict()['state'].values()},{2.})
if __name__=='__main__':unittest.main()
