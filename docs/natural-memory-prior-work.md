# Reading notes: memory through a learning encoder

Search date 2026-11-03. Reading for [memory on natural images](natural-memory-v1-pilot.md), and part of the reading owed by the [image-free](sense-image-free-v1-pilot.md) and [slow-average](sense-slow-average-v1-pilot.md) pilots. Sources and queries: arXiv pages and PDFs below; web searches for SimCLR kNN on CIFAR-10, exemplar-free drift compensation, self-healing codes, online continual self-supervised learning and expert networks. Tags as in [prior work](prior-work.md).

## Sources

**He et al. 2019, Momentum Contrast ([arXiv 1911.05722](https://arxiv.org/abs/1911.05722)). [read: sections 3.2, 3.3, 4.1 ablation]**
- *Demonstrated:* on ImageNet linear evaluation, with a queue of 4,096 keys, a key encoder averaged with m = 0, 0.9, 0.99, 0.999, 0.9999 gives failure, 55.2, 57.8, 59.0, 58.9%. A memory bank whose keys come from many past encoders reaches 58.0% with 65,536 negatives and is 2.6% worse than MoCo at that size. A plain copy of the working encoder fails to train.
- *Setting differs:* keys there are negatives in a training loss, reused for one queue's length, and old keys are thrown away. Here keys are memories to be kept and read.
- *Hypothesis drawn:* keys from nearby encoder versions are consistent enough. It does not say that a key lasts longer than the queue. The authors remove the oldest keys because they are "the least consistent".

**Yu et al. 2020, Semantic Drift Compensation ([arXiv 2004.00440](https://arxiv.org/abs/2004.00440)). [read: sections 3–5.1]**
- *Demonstrated:* class prototypes in an embedding network can be kept without exemplars by moving each prototype by a Gaussian-weighted mean of how current-task inputs moved (eq. 10–11; σ = 0.3, or 0.2 on CIFAR-100). This improves finetuning by 11.9% on CUB and 6.2% on Flowers after six tasks. Embedding networks forget much less than softmax networks. The same paper's fixed-after-first-task baseline, E-Fix, gives 50.3 on CUB and 76.9 on Flowers, against 49.3 and 35.3 for finetuning with compensation.
- *Setting differs:* supervised metric learning at a low learning rate (1e-5 or 1e-6), with drift far smaller than here. Prototypes are one per class in 512 dimensions, not many units per fact.
- *Hypothesis drawn:* this is the established form of what the [image-free pilot](sense-image-free-v1-pilot.md) tried with a global linear map, and it was not cited there. So it is tested here as an arm. A frozen encoder beating compensation is also an existing result, not ours alone.

**Rule and O'Leary 2022, Self-healing codes ([PMC8851551](https://pmc.ncbi.nlm.nih.gov/articles/PMC8851551)). [read through a summarizing tool, so abstract-level confidence]**
- *Demonstrated in simulation:* a Hebbian and homeostatic readout can track a drifting population code without external error, if tuning stays redundant and drift keeps the coarse geometry of the variable. Hebbian homeostasis alone eventually forgets the original code; internal models need occasional rehearsal.
- *Setting differs:* abstract rate networks with random-walk drift on 100 units; not learned features, not arbitrary one-off facts.
- *Hypothesis drawn:* tracking depends on what is preserved under drift. Contrastive learning from a glyph start changes the geometry, so this source predicts no self-healing there.

**Aljundi et al. 2017, Expert Gate ([arXiv 1611.06194](https://arxiv.org/abs/1611.06194)). [abstract, with search summary]**
- *Demonstrated:* one expert network per task, with a small autoencoder per task selecting which expert to run at test time, so one expert need be in memory at a time. No earlier data stored.
- *Setting differs:* tasks are told at training time; supervised classification and video prediction.
- *Hypothesis drawn:* the `versions` arm is a network-per-era design in this family. Its cost at read time, one forward pass per copy, is what a gate would cut. It is not new.

**Rusu et al. 2016, Progressive Neural Networks ([arXiv 1606.04671](https://arxiv.org/abs/1606.04671)). [abstract]**
- *Demonstrated:* on reinforcement learning tasks, a network that adds a column per task, with old columns frozen, is "immune to forgetting" and reuses features through lateral connections.
- *Hypothesis drawn:* versions without sharing are the plainest case; lateral sharing is a route to cheaper copies.

**Hayes et al. 2019, REMIND ([arXiv 1910.02509](https://arxiv.org/abs/1910.02509)). [abstract]**
- *Demonstrated:* replay from compressed representations outperforms other incremental class learning methods on ImageNet.
- *Hypothesis drawn:* if rehearsal is wanted without images, codes from a frozen early stage are the established way, though that needs a stable early stage.

**Online continual self-supervised learning: SCALE ([CVPRW 2023](https://openaccess.thecvf.com/content/CVPR2023W/CLVision/html/Yu_SCALE_Online_Self-Supervised_Lifelong_Learning_Without_Prior_Knowledge_CVPRW_2023_paper.html)), CaSSLe, CLIMB ([code](https://github.com/lefebvju/climb)). [search summary only]** Single-pass, non-iid streams with no task labels; distillation or stored subsets. Read before any contract on a shifting stream.

**CIFAR-10 kNN range.** t-SimCLR ([arXiv 2205.14814](https://arxiv.org/pdf/2205.14814)) reports SimCLR at 87.2% nearest-neighbour accuracy after 200 epochs on a ResNet-18 **[search summary]**. Different protocol and a much larger network; context only.

## What this predicts for the pilot

- A fast-moving encoder cannot keep old keys readable: MoCo discards them for that reason **[read]**.
- Compensation depends on nearby inputs moving as stored points move. Here the stream is the same kind of image as memory, which is its best case; across a change of kind it has no nearby inputs **[derived]**.
- A frozen copy per era cannot forget and costs linear growth in copies and reads **[derived]**. Prior work names the open problems: which copy to run, and when to add one.
- A glyph-trained network maps natural images near each other: nothing it was trained on required telling them apart **[derived, and measured in the pilot]**.

## Follow-up: train for address compatibility

Search date 2026-11-03. Queries: “backward compatible representation learning Shen 2020 old embeddings new model compatibility”; “CaSSLe self supervised continual learning feature distillation predictor frozen representations”. These are candidate directions, not a contract.

- **Shen et al. 2020, Towards Backward-Compatible Representation Learning ([primary abstract](https://arxiv.org/abs/2003.11942)). [abstract]** Demonstrates new query embeddings compatible with old gallery embeddings in face recognition, without rebuilding the gallery. Different setting: supervised recognition and model upgrades, not an unlabelled changing stream with arbitrary fact associations. **[hypothesis]** Address compatibility should be a training objective, rather than something repaired only after encoder updates. The source does not establish feasibility under this pilot's restrictions.
- **Fini et al., Self-Supervised Models are Continual Learners / CaSSLe ([primary abstract](https://arxiv.org/abs/2112.04215), [official implementation](https://github.com/DonkeyShot21/cassle)). [abstract]** Uses a predictor from current representations to past representations for self-supervised distillation and reports improved continual representation learning. **[derived distinction]** Retaining representation quality does not itself establish retrieval against immutable historical keys. **[hypothesis]** A compatibility head is worth considering, but current-domain distillation alone may leave absent old domains unconstrained. Read the full methods before specifying an arm.

**Program implications [hypotheses, not demonstrated]:** first test whether explicitly trained compatibility can preserve old-key retrieval while new-view performance improves. Use a small fixed budget of representative anchor inputs as a diagnostic control, disclose retained bytes, and compare with a zero-anchor arm. If the anchor control works but the zero-anchor arm fails, investigate compressed or generative rehearsal rather than treating drift compensation as sufficient. A stable low-level representation with a plastic reader is another candidate; it must retain information useful for later distinctions, and freezing it alone does not protect the reader from forgetting. No new experiment or performance claim is established here.
