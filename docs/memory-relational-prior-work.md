# Memorized facts, relational inference, and mixed sparsity

Search date: 2026-10-30. Targeted reading for the arbitrary-pairings discussion;
not an exhaustive survey or novelty assessment. Queries covered DNC graph inference,
complementary structure-learning, Tolman–Eichenbaum, Go-CLS, Wide & Deep,
end-to-end memory networks, and shared/routed experts.

## Directly relevant studies

- **Tolman–Eichenbaum Machine (Whittington et al., 2020)**
  [source](https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106/) **[read: summary and results excerpt]**.
  Separates transferable relational structure from sensory observations and binds
  them in relational memory; demonstrates structural generalization across environments.
  Difference: graph-transition prediction, not our mixed reader.
  **[hypothesis]** Test inference involving newly memorized symbol–count bindings,
  in addition to recall and preservation of old structured tasks.
- **Complementary Structure-Learning Neural Networks for Relational Reasoning
  (Russin et al., 2021)** [source](https://pmc.ncbi.nlm.nih.gov/articles/PMC8491570/)
  **[read: introduction, modeling framework, results and discussion]**.
  Models transitive inference through learned cortical maps or retrieval of episodic
  key–value memories. The two models are tested separately, not as our joint architecture.
  The episodic model uses one-hot identities and meta-training on related inference tasks;
  the cortical model processes face images. **[hypothesis]** Recognition-controlled
  inputs can isolate binding and inference from visual recognition in our diagnostic.
- **Differentiable Neural Computer (Graves et al., 2016)**
  [source](https://www.nature.com/articles/nature20101) **[abstract]**.
  Neural controller plus external read/write memory learns graph paths and missing-link
  inference, including transport and family-tree examples. Difference: facts are held in
  external memory, not exclusively in learned reader weights. **[analogy]** A task can
  require both newly supplied facts and a reusable inference procedure.
- **End-To-End Memory Networks (Sukhbaatar et al., 2015)**
  [source](https://arxiv.org/abs/1503.08895) **[abstract]**.
  Multiple attention hops over external memory improve reported question-answering results.
  Difference: external facts and learned retrieval, not dense/sparse synaptic storage.
  **[analogy]** Retrieval and inference can cooperate without consolidation first.

## Mixing systems and meanings of sparsity

- **Organizing memories for generalization in complementary learning systems
  (Sun et al., 2023)** [source](https://www.nature.com/articles/s41593-023-01382-9)
  **[read: indexed model and results excerpts; not full text]**.
  A linear student learns with a sparse Hopfield notebook. Limited predictability makes
  excessive consolidation harmful to generalization in the model. Difference: analytical
  teacher–student setting, not relational reader training. **[hypothesis]** Specific
  unpredictable associations can retain value in memory without being consolidated.
- **Wide & Deep (Cheng et al., 2016)** [source](https://arxiv.org/abs/1606.07792)
  **[abstract]**. Joint wide feature-cross model and deep embedding network combine
  memorization and generalization, improving recommendation over either alone.
  Difference: sparse input/features, not RigL sparse connectivity; no relational proof task.
  **[analogy]** Joint prediction can preserve exceptions alongside shared regularities.
- **DeepSeekMoE (Dai et al., 2024)** [source](https://arxiv.org/abs/2401.06066)
  **[abstract]**. Combines shared experts and sparsely selected routed experts for language
  modeling. Difference: conditional activation of experts, which can themselves be dense;
  no demonstrated facts-versus-reasoning division. **[analogy]** Shared and specialized
  computation can coexist, but this does not validate our 48/16 connectivity split.

## Implications for this program

**[derived distinctions]** Sparse weights, sparse activations, sparse feature vectors,
and sparse expert routing are different interventions. Complementary-learning accounts
often associate sparse pattern-separated representations with episodic memory and
overlapping representations with integration. This neither predicts nor contradicts
our low-fan-in reader result: the sparsity constraints differ.

**[hypothesis]** After learning arbitrary bindings X→3 and Y→5, test an untaught comparison
X<Y as well as direct recall. Counterbalance assignments, withhold these comparisons from
training, and score recognition, recall, and inference separately. This tests cooperation
between remembered facts and learned relations; the current recall/old-transfer endpoints
primarily test their coexistence. No contract change or experiment is authorized by these notes.

## Routing and retention mechanisms (discussion follow-up)

Search date: 2026-10-30; queries: LSTM forget gates, sparsely gated mixture of experts,
Titans adaptive forgetting, elastic weight consolidation.

- [Gers et al., 2000](https://pubmed.ncbi.nlm.nih.gov/11032042/)
  **[abstract]**: adaptive forget gates reset recurrent memory cells. Difference:
  activation-state retention, not protection of associations in shared weights.
- [Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)
  **[abstract]**: a trainable gate selects sparse combinations of experts. Difference:
  demonstrated conditional computation, not an identified memory/reasoning division.
- [Titans, 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/a4ca07aa108036f80cbb5b82285fd4b1-Paper-Conference.pdf)
  **[read: indexed mechanism excerpts; not full text]**: online neural associative memory
  uses a surprise-related update and adaptive forgetting. Difference: contextual sequence
  memory, not demonstrated lifetime preservation of arbitrary multimodal facts.
- [Kirkpatrick et al., 2017](https://doi.org/10.1073/pnas.1611835114)
  **[abstract]**: elastic weight consolidation constrains changes to parameters important
  for earlier tasks. Difference: parameter protection, not a semantic fact-level lifetime.

**[hypothesis]** Use separately controlled reads, writes, and retention in jointly trained
modules. Start with two fixed memory timescales and learn access before learning decay.
Train gates on delayed queries, including rare useful facts, so immediate prediction
error alone does not decide retention. Gating shared parameters cannot guarantee selective
fact deletion or preservation; dedicated associative state makes that experiment clearer.
