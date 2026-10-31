# Established remedies relevant to forgetting mechanism v1

Search date: 2026-10-31. Queries: Davari representation forgetting; Dark Experience
Replay; Learning without Forgetting; REMIND frozen layers and replay. This is a
targeted comparison, not a comprehensive survey or new experiment contract.

- [Davari et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Davari_Probing_Representation_Forgetting_in_Supervised_and_Unsupervised_Continual_Learning_CVPR_2022_paper.html)
  **[read: abstract and introduction excerpt]**: old-task performance can collapse
  despite useful representations surviving; feature permutation illustrates readout
  incompatibility. Difference: classification benchmarks, not symbol–count matching.
  **[analogy]** Our frozen/rebuilt-bank and probe results fit an established distinction,
  rather than establishing a new category of forgetting.
- [Buzzega et al., NeurIPS 2020](https://arxiv.org/abs/2004.07211)
  **[abstract; introduction excerpt also read]**: Dark Experience Replay combines
  rehearsal with matching historical logits. Difference: a stored-example buffer,
  rather than memory exclusively in network weights. **[hypothesis]** Compare bounded
  replay with replay plus preservation of past matching scores, at declared memory
  and compute budgets. Replaying the whole accessible rendering pool is not the same
  storage constraint as a bounded buffer.
- [Li and Hoiem, Learning without Forgetting](https://arxiv.org/abs/1606.09282)
  **[abstract]**: preserves old capabilities while training on new-task data without
  old training data. Difference: their vision tasks and data support differ from ours.
  **[hypothesis]** Constraints evaluated only on context inputs may fail to protect
  arbitrary symbol associations absent from those inputs; do not assume transfer.
- [Hayes et al., REMIND, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530460.pdf)
  **[read: method excerpt, section 3]**: freezes early layers and replays compressed
  intermediate features to train plastic layers. Difference: feature replay requires
  retained state and an intentionally stable feature interface. **[hypothesis]** A
  stable encoder plus associative replay is a relevant combined control; freezing
  alone did not preserve our reader, so REMIND must not be represented as that test.

**Program implication [hypothesis]:** prioritize established retention baselines and
their retention–new-learning–compute–storage tradeoffs before another broad diagnosis
study. A first-moment reset remains a focused mechanistic diagnostic; the zero-gradient
result does not establish it as a remedy during actual context training.
