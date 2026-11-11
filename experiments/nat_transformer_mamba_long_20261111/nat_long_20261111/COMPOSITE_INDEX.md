# Post-hoc composite: explicit preferences, not intelligence



Under the disclosed default, Transformer = 100 and Mamba = 42.51. This ranking favors Transformer for a workload equally weighting relative quality and training throughput. It is not a universal architecture winner.



## Normalization and weights

Transformer=100 reference. Text uses the ratio of geometric mean probability assigned to correct target bytes, 2**(bpb_T-bpb_M). Tasks use ratios of held-out per-byte accuracy. Ratios are geometrically averaged across paired seeds. No arbitrary min-max scaling, clipping, pseudocounts, benchmark-standard or intelligence interpretation.

R=100*exp(alpha*(beta*mean_seed(mean_text[-ln(2)*(bpb_M-bpb_T)])+(1-beta)*mean_seed(mean_task[ln(acc_M/acc_T)]))+(1-alpha)*ln(median_step_ms_T/median_step_ms_M)); alpha=beta=0.5 by assistant default, not user-stated preferences.

Defaults are assistant-chosen for this requested comparison, not learned user preferences. Quality receives 50% and standalone training throughput 50%. Within quality, text gets 50% (two corpora equally) and adapted tasks get 50% (four tasks equally). The earlier full report and all component metrics are unchanged.



Quality-only index at the same text/task weights: Mamba 91.14, Transformer 100. Relative training throughput: 0.1982 for Mamba versus Transformer.



## Components

- wikitext: Mamba/Transformer ratio 1.06996; quality weight 0.250; all paired seed ratios [1.07660451155243, 1.0558577445116457, 1.0775697002828384].

- fineweb: Mamba/Transformer ratio 1.04343; quality weight 0.250; all paired seed ratios [1.0474528090814177, 1.0348332291357591, 1.0480558184735496].

- copy: Mamba/Transformer ratio 0.26874; quality weight 0.125; all paired seed ratios [0.3958047082531831, 0.18939972506491526, 0.25890076335877865].

- reverse: Mamba/Transformer ratio 1.12973; quality weight 0.125; all paired seed ratios [1.412128017967434, 0.7619877942458587, 1.3399898459976307].

- sort: Mamba/Transformer ratio 1.08414; quality weight 0.125; all paired seed ratios [0.9862015284460798, 1.1392362536862144, 1.1341665135085461].

- prefixsum_mod10: Mamba/Transformer ratio 1.16061; quality weight 0.125; all paired seed ratios [1.2315476190476191, 1.2245687091017252, 1.0366363412835014].



## Uncertainty

Default Mamba score by paired seed with timing fixed: [44.1157091627422, 40.72756352011932, 42.74113411310757]. Conditional empirical 95% bootstrap interval: [40.766373578879474, 44.179085215062365].

5000 resamples of three paired fitted seeds and 40 paired standalone timing trials. Conditional on fixed evaluation inputs and this host/session. This does not include fresh data/domain sampling, training hyperparameter uncertainty, other hardware or many independent model fits; three seeds are weak evidence for population inference.



## Sensitivity and matched time

With no speed weight, Mamba overtakes Transformer when text receives more than 81.37% of quality weight. With text-only quality, Mamba wins only when quality receives more than 96.71% and speed receives the remainder. These thresholds reflect this particular ratio normalization, not a universal utility tradeoff.

At 900 training CPU seconds, the separate text-only index is Mamba 102.25, Transformer 100: Mamba's text prediction is better at matched compute time despite slower optimizer steps. Throughput and learning efficiency are different.

- default_without_prefix_sum_task: Mamba 41.15, Transformer 100.

- default_using_task_correct_byte_log_probability_instead_of_accuracy: Mamba 40.97, Transformer 100.

- default_using_full_18000_update_cpu_throughput: Mamba 45.99, Transformer 100.

The complete two-weight sensitivity grid is in composite_sensitivity.json; axes cover every integer percentage from 0 to 100. This is analytic reweighting of existing measurements, not new model training.



## Conditions and limits

- Quality combines the 16000-update text-only checkpoint and separate 2000-update task-adapted checkpoint for each paired seed. It is a multi-workload workflow index, not one checkpoint simultaneously evaluated on every task.

- The default speed component is the standalone median whole-model training step with equal batch and shapes. It is not time to a target quality, learning efficiency, full training cost, or monetary utility.

- The 900-CPU-second comparison is kept separate and contains text only. No speed multiplier is added to that already time-matched score.

- Exact-sequence accuracy and task loss remain in the full report; exact match is not included again as an overlapping metric. Zero exact match on reverse/prefix-sum and shortcut-level prefix-sum byte accuracy prevent an arithmetic or intelligence claim.

- A large throughput ratio can dominate modest predictive-probability ratios. Equal weights on log ratios encode that preference; other transformations and task choices are also judgments.

- No universal overall winner follows. Score differences are not percentages of intelligence, economic value, or frontier-scale benefit.
