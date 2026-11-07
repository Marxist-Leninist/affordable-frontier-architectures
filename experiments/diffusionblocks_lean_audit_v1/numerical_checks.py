"""Bounded independent checks. No paper code is imported or executed."""
from fractions import Fraction as Q
import json
import numpy as np

rng = np.random.default_rng(20261107)

def corrected(z, d, old, nxt):
    return z + (nxt-old)*(z-d)/old

def printed(z, d, old, nxt):
    return z + (old-nxt)*(z-d)/old

z, d, old, nxt = Q(1), Q(0), Q(1), Q(1, 2)
assert corrected(z,d,old,nxt) == Q(1,2)
assert printed(z,d,old,nxt) == Q(3,2)

# Time-varying affine denoisers, checked independently against product-sum.
# M[k]=(1-alpha[k])*I+alpha[k]*A[k], c[k]=alpha[k]*b[k].
matrix_errors = []
for trial in range(100):
    dim, steps = 4, 8
    alphas = rng.uniform(.05,.6,steps)
    As = rng.normal(0,.2,(steps,dim,dim))
    bs = rng.normal(0,.3,(steps,dim))
    z0 = rng.normal(size=dim)
    Ms = [(1-a)*np.eye(dim)+a*A for a,A in zip(alphas,As)]
    cs = [a*b for a,b in zip(alphas,bs)]
    state = z0.copy()
    for alpha,A,b in zip(alphas,As,bs):
        denoised = A@state+b
        state = state+alpha*(denoised-state)
    product = np.eye(dim)
    for M in Ms:
        product = M@product
    closed = product@z0
    for k,c in enumerate(cs):
        tail = np.eye(dim)
        for M in Ms[k+1:]:
            tail = M@tail
        closed = closed+tail@c
    matrix_errors.append(float(np.max(np.abs(state-closed))))
assert max(matrix_errors) < 1e-12

# A one-step structural equivalence is valid for nonlinear maps too,
# but this remains an evaluation of the original nonlinear function.
z = rng.normal(size=20)
alpha = .25
residual = np.tanh(z)+.1*z*z
denoiser = z+residual/alpha
reparam_error = float(np.max(np.abs((z+residual)-(z+alpha*(denoiser-z)))))
assert reparam_error < 1e-12

loss = np.array([1.,3.])
weights = np.array([1.,3.])
intended = float(np.mean(loss*weights))
broadcast = float(np.mean(loss*weights[:,None]))
assert intended == 5 and broadcast == 4

report = {
    'kind': 'independent exact arithmetic and numerical identities, not a language benchmark',
    'seed': 20261107,
    'paper_sign_counterexample': {'correct': '1/2', 'printed': '3/2'},
    'affine_matrix_cases': 100,
    'dimension': 4,
    'steps': 8,
    'max_affine_matrix_absolute_error': max(matrix_errors),
    'nonlinear_one_step_reparameterization_absolute_error': reparam_error,
    'weighted_loss_intended': intended,
    'weighted_loss_broadcast': broadcast,
    'all_assertions_passed': True,
}
print(json.dumps(report,indent=2))
