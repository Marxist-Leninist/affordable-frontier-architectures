import Init

/-
Algebraic checks for DiffusionBlocks, arXiv:2506.14202v4, equations 1, 3--6,
and Figure 3. No assertion about generalization, optimizer convergence,
wall-clock costs, stochastic processes, or floating-point execution is made.

The generic results use Lean's standard Field/CommRing laws, not custom axioms.
All concrete counterexamples use exact rational numbers.
-/
namespace DiffusionEquivalence
open Lean.Grind
set_option maxHeartbeats 100000

def corrected {K : Type} [Field K] (z d old next : K) : K :=
  z + (next-old)*((z-d)/old)

def printed {K : Type} [Field K] (z d old next : K) : K :=
  z + ((old-next)/old)*(z-d)

-- Euler from dz/dsigma = -sigma*score, with score=(d-z)/sigma^2.
theorem euler_from_score {K : Type} [Field K]
    (z d old next : K) (h : old ≠ 0) :
    z + (next-old)*(-old*((d-z)/(old*old))) = corrected z d old next := by
  unfold corrected
  grind

-- The step-size is signed: next-old, not the positive old-next.
theorem euler_residual {K : Type} [Field K] (z d old next : K) :
    corrected z d old next = z + ((old-next)/old)*(d-z) := by
  unfold corrected
  grind

theorem residual_mixture {K : Type} [CommRing K] (z d a : K) :
    z+a*(d-z) = (1-a)*z+a*d := by grind

-- An arbitrary residual function can be represented at a nonzero step size.
-- This proves equality of the update map, NOT equality of training objectives.
theorem residual_reparameterization {K : Type} [Field K]
    (z f a : K) (ha : a ≠ 0) :
    z+a*((z+f/a)-z) = z+f := by grind

theorem printed_sign_counterexample :
    corrected (1 : Rat) 0 1 (1/2) = 1/2 ∧
    printed (1 : Rat) 0 1 (1/2) = 3/2 ∧
    corrected (1 : Rat) 0 1 (1/2) ≠ printed (1 : Rat) 0 1 (1/2) := by
  decide +kernel

-- Perfect denoising of a fixed target contracts the residual by next/old.
theorem ideal_denoiser {K : Type} [Field K]
    (z y old next : K) (h : old ≠ 0) :
    corrected z y old next = y+(next/old)*(z-y) := by
  unfold corrected
  grind

theorem ideal_two_steps_telescope {K : Type} [Field K]
    (z y s0 s1 s2 : K) (h0 : s0 ≠ 0) (h1 : s1 ≠ 0) :
    corrected (corrected z y s0 s1) y s1 s2 = y+(s2/s0)*(z-y) := by
  unfold corrected
  grind

theorem terminal_perfect_denoiser {K : Type} [Field K]
    (z y old : K) (h : old ≠ 0) : corrected z y old 0 = y := by
  unfold corrected
  grind

-- D(z)=a*z+b induces an affine update m*z+c.
theorem affine_denoiser_update {K : Type} [CommRing K] (z a b step : K) :
    z+step*((a*z+b)-z) = (1-step+step*a)*z+step*b := by grind

theorem two_affine_steps {K : Type} [CommRing K] (m1 c1 m2 c2 z : K) :
    m2*(m1*z+c1)+c2 = (m2*m1)*z+(m2*c1+c2) := by grind

def affineIter {K : Type} [CommRing K] (m c z0 : K) : Nat → K
  | 0 => z0
  | n+1 => m*affineIter m c z0 n+c

-- The n-term geometric sum: 1+m+...+m^(n-1).
def geometric {K : Type} [CommRing K] (m : K) : Nat → K
  | 0 => 0
  | n+1 => geometric m n + m^n

theorem geometric_identity {K : Type} [CommRing K] (m : K) (n : Nat) :
    (m-1)*geometric m n = m^n-1 := by
  induction n with
  | zero => simp [geometric, Semiring.pow_zero]; grind
  | succ n ih =>
    simp only [geometric, Semiring.pow_succ]
    grind

theorem affine_closed_sum {K : Type} [CommRing K] (m c z0 : K) (n : Nat) :
    affineIter m c z0 n = m^n*z0 + geometric m n*c := by
  induction n with
  | zero => simp [affineIter, geometric, Semiring.pow_zero]; grind
  | succ n ih =>
    simp only [affineIter, geometric, Semiring.pow_succ, ih]
    have hg := geometric_identity m n
    grind

-- Genuine non-recursive closed form for the time-homogeneous affine case.
theorem affine_closed_fraction {K : Type} [Field K]
    (m c z0 : K) (n : Nat) (hm : m ≠ 1) :
    affineIter m c z0 n = m^n*z0 + ((m^n-1)/(m-1))*c := by
  have hg := geometric_identity m n
  have hf := affine_closed_sum m c z0 n
  grind

-- A local block objective ignores other parameter blocks by construction.
theorem local_objective_independent {I P : Type}
    (loss : P → Rat) (theta theta' : I → P) (b : I)
    (h : theta b = theta' b) : loss (theta b) = loss (theta' b) := by rw [h]

-- A minimal exact counterexample to equality of local and end-to-end losses.
-- Both local denoisers are constants, target=1, initial z=0, step sizes=1/2.
def localLoss (a b : Rat) : Rat := (a-1)^2+(b-1)^2
def twoStepOutput (a b : Rat) : Rat := (a/2)/2+b/2
def endToEndLoss (a b : Rat) : Rat := (twoStepOutput a b-1)^2

theorem training_losses_not_identical :
    localLoss 1 1 = 0 ∧ endToEndLoss 1 1 = 1/16 ∧
    localLoss 2 1 = 1 ∧ endToEndLoss 2 1 = 0 := by
  decide +kernel

theorem local_loss_nonnegative (a b : Rat) : 0 ≤ localLoss a b := by
  unfold localLoss
  have ha : (0 : Rat) ≤ (a-1)^2 := OrderedRing.sq_nonneg
  have hb : (0 : Rat) ≤ (b-1)^2 := OrderedRing.sq_nonneg
  grind

theorem endpoint_loss_nonnegative (a b : Rat) : 0 ≤ endToEndLoss a b := by
  exact OrderedRing.sq_nonneg

-- In the public ViT code, [N] times [N,1] broadcasts to [N,N].
-- For two examples, the resulting average can differ from a weighted mean.
theorem weighted_broadcast_counterexample :
    ((1 : Rat)*1+3*3)/2 = 5 ∧
    ((1 : Rat)+3)/2*((1+3)/2) = 4 := by decide +kernel

-- Figure 3's literal i=0,...,T-1 loop does not assign state z_T.
theorem displayed_loop_excludes_output (T i : Nat) (hi : i < T) : i ≠ T := by omega

theorem displayed_first_index_is_negative : (0 : Int)-1 = -1 := by omega

#print axioms euler_from_score
#print axioms euler_residual
#print axioms residual_mixture
#print axioms residual_reparameterization
#print axioms printed_sign_counterexample
#print axioms ideal_denoiser
#print axioms ideal_two_steps_telescope
#print axioms terminal_perfect_denoiser
#print axioms affine_denoiser_update
#print axioms two_affine_steps
#print axioms geometric_identity
#print axioms affine_closed_sum
#print axioms affine_closed_fraction
#print axioms local_objective_independent
#print axioms training_losses_not_identical
#print axioms local_loss_nonnegative
#print axioms endpoint_loss_nonnegative
#print axioms weighted_broadcast_counterexample
#print axioms displayed_loop_excludes_output
#print axioms displayed_first_index_is_negative
end DiffusionEquivalence
