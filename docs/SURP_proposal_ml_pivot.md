# Prompt Representation Learning for LLM-Assisted Software Development

Medha Bhattacharya | mbhatta1@uci.edu | B.S. Computer Science, UCI, Class of 2027  
Proposed: Summer 2026 | UCI Summer Undergraduate Research Program (SURP)

## Introduction

Large language models have become part of the day-to-day workflow of software development, but there is still very little empirical evidence about how developers actually structure prompts in real repositories and whether different prompt styles lead to measurably different outcomes. Most existing work studies LLM coding tools in controlled settings or focuses on usability and trust. Far less is known about the prompt representations themselves: how they cluster, how they evolve over time, and whether they can be modeled using modern machine learning methods.

This project reframes the problem as a small-data ML task. Instead of trying to build or fine-tune a large generative model from scratch, I will construct a curated dataset of real-world LLM-assisted development prompts and use pretrained embeddings, weak supervision, clustering, and lightweight predictive models to study prompt behavior. The core research question is whether prompt text and its surrounding context contain enough signal to learn meaningful representations of prompting strategy and to predict downstream quality-related outcomes.

The project is intentionally designed around the constraints of the available data. My current pilot dataset is limited in size and noisy by nature, which makes large-scale fine-tuning unrealistic as the primary approach. That limitation is actually a strength for this proposal: it motivates a rigorous small-data ML pipeline centered on representation learning, model comparison, and careful validation rather than overclaiming about large-model training.

## Background

### LLMs in Software Development

Tools like GitHub Copilot, Cursor, and ChatGPT are increasingly used to generate code, explain errors, and assist with refactoring. While prior research has explored these tools from a human-computer interaction perspective, the machine learning side of the problem remains underexplored: what can we learn from the prompts developers write, and can those prompts be modeled in a way that reveals useful structure?

### Prompt Representation Learning

In natural language processing, pretrained encoders and sentence embeddings often provide a strong foundation for small-data tasks. Prompt text is a good candidate for this approach because it is short, semantically dense, and often repetitive across projects. By embedding prompts and grouping them into latent categories, I can study whether real-world prompting behavior exhibits consistent structure that can be learned without training a large model end to end.

### Weak Supervision and Small-Data Learning

Because prompt labels are not directly available at scale, the project will rely on weak supervision and manual validation. Signals such as inline prompt comments, repository metadata, commit messages, and surrounding code context can be used to create noisy labels and support small supervised experiments. This is a natural fit for a research setting where the goal is not just to maximize accuracy, but to understand which ML methods remain robust when labels are sparse and imperfect.

## Project Overview and Objectives

This project has four primary objectives:

1. Build a curated dataset of real-world LLM-assisted development prompts and their surrounding context.
2. Learn a taxonomy of prompt styles using pretrained embeddings, clustering, and manual validation.
3. Model how prompt representations and categories evolve over time within repositories.
4. Evaluate whether prompt patterns are predictive of lightweight quality signals and other downstream outcomes.

The emphasis is on representation learning and predictive modeling under limited supervision. The output will be both a reusable dataset and a set of empirical findings about which ML methods are most informative for prompt analysis in the wild.

## Methodology

### 1. Repository Mining and Dataset Construction

The foundation of the project is a dataset of candidate repositories that show evidence of LLM-assisted development. I will continue using the current repository mining pipeline to search for signals such as explicit references to Copilot, Cursor, or ChatGPT; `.cursor` or `.copilot` configuration files; prompt-like inline comments; and relevant topic tags. Each candidate repository will be deduplicated and filtered, producing a corpus of prompt-bearing examples with repository-level metadata.

For each qualifying repository, I will extract prompt text when available, nearby code context, commit timestamps, language, repository size, star count, and lightweight indicators of revision behavior. Where available, I will also record versioned snapshots of the prompt and its associated code diff so that prompt changes can be analyzed as a sequence rather than a static string.

Because the mined data is noisy, I will create a weak labeling scheme instead of assuming all mined examples are equally reliable. For example, inline prompt comments can serve as high-confidence prompt instances, while README references or commit messages can serve as lower-confidence supporting signals. A manually reviewed subset will be used to validate label quality and refine the taxonomy.

### 2. Prompt Embeddings and Taxonomy Development

The first ML task is to build a prompt representation space. I will embed prompts using a pretrained sentence embedding model such as SBERT or a comparable open encoder. These embeddings will be the primary representation for all downstream tasks.

I will then compare unsupervised and semi-supervised methods for discovering prompt categories, including:

- k-means clustering
- hierarchical clustering
- dimensionality reduction for visualization
- topic-model-style exploration of embedding neighborhoods

Each cluster will be manually inspected and assigned a semantic label. Based on prompt engineering literature and the structure of real prompts, I expect categories such as:

- imperative task prompts
- context-rich specification prompts
- debugging and correction prompts
- refactoring prompts
- test-generation prompts
- architectural or design prompts

The goal here is not just taxonomy creation, but representation analysis: I want to determine whether pretrained embeddings naturally separate these prompting behaviors and how stable those separations are across projects.

### 3. Lightweight Predictive Modeling

Once the taxonomy exists, I will train lightweight models on top of the prompt embeddings to test whether prompt style is predictive of downstream signals. Candidate tasks include:

- prompt category classification
- prompt clarity or specificity scoring
- prediction of whether a prompt is followed by substantial manual revision
- prediction of lightweight code-quality proxies when available

The initial models will be simple and interpretable, such as logistic regression, random forest, or gradient-boosted models trained on embeddings plus metadata features. This gives a strong baseline and avoids overfitting on a limited dataset.

If the labeled prompt set is large enough, I will add one parameter-efficient fine-tuning experiment using LoRA or a similar PEFT method on a small open model. The purpose of that experiment is not to claim that full-scale fine-tuning is feasible, but to test whether adapting a compact encoder yields measurable gains over frozen embeddings on prompt classification. That makes the ML contribution stronger while remaining realistic for a summer project.

### 4. Longitudinal Evolution Analysis

Because the data is timestamped, I can model prompt behavior as a time sequence rather than a one-off classification task. I will track how prompt categories and embedding distributions change over time within repositories, looking for:

- shifts toward more context-rich prompts
- convergence on shared team prompt conventions
- changes in prompt length or specificity
- transitions between prompt categories as a project matures

This analysis will be done with embedding trajectories, cluster transition matrices, and simple temporal summaries. If the data permits, I will compare early vs. late repository phases to see whether prompt patterns become more specialized over time.

### 5. Outcome Analysis and Model Evaluation

The downstream outcome modeling will be intentionally modest and tied to what the dataset can actually support. Instead of promising full code-quality ground truth, I will use lightweight and observable signals such as:

- revision intensity after a prompt-bearing commit
- commit message markers associated with bug fixing or refactoring
- static-analysis complexity measures where code is available
- PR review timing and merge behavior when accessible

These outcomes are useful as exploratory targets, but they are not guaranteed to be available for every repository. For that reason, the central deliverable will be the prompt representation and classification work; the outcome modeling will be presented as an applied validation layer rather than the entire project.

Model evaluation will include:

- clustering quality metrics
- classification metrics such as macro F1 and accuracy
- ablation studies comparing text-only, metadata-only, and combined feature sets
- qualitative review of representative examples from each cluster

## Why This Is an ML Project

This pivot makes the project materially more machine-learning-centered in four ways:

1. The core artifact is a learned prompt representation space.
2. The main experimental question is how well pretrained and lightweight models classify or separate prompt styles.
3. Weak supervision and PEFT are explicit methods in the design, not afterthoughts.
4. The evaluation focuses on model behavior under noisy, real-world data rather than on manual descriptive analysis alone.

In other words, the project is no longer mainly about mining software repositories for signals. It is about using ML to discover structure in real developer prompts and to test whether that structure generalizes to downstream signals.

## Detailed Plan and Timeline

### Weeks 1-2: Literature Review and Dataset Definition

- Review work on prompt engineering, sentence embeddings, weak supervision, and PEFT.
- Finalize the prompt extraction and labeling scheme.
- Expand the repository mining pipeline if needed to support prompt-level extraction.

### Weeks 3-4: Data Collection and Annotation

- Mine candidate repositories and assemble the prompt dataset.
- Manually annotate a sample of prompts for taxonomy development.
- Create a weak-labeling rubric for high-confidence and low-confidence examples.

### Weeks 5-6: Embeddings and Clustering

- Compute prompt embeddings using a pretrained encoder.
- Run clustering experiments and inspect cluster quality.
- Refine the taxonomy through manual validation and example-based labeling.

### Weeks 7-8: Predictive Modeling

- Train lightweight classifiers and regressors on top of embeddings.
- Compare text-only, metadata-only, and combined baselines.
- Start a temporal analysis of prompt drift and category transitions.

### Weeks 9-10: Optional PEFT and Reporting

- If the dataset supports it, run one LoRA/PEFT experiment on a compact model.
- Evaluate whether fine-tuning improves prompt classification over frozen embeddings.
- Write the final report and prepare the presentation with emphasis on ML findings and limitations.

## Proposed Budget

Funding is requested for computational resources needed for embedding inference, model comparison, and a small optional fine-tuning experiment.

| Item | Rate | Total |
| --- | --- | --- |
| Cloud compute for embedding inference and model training | Estimated 200 hours on a modest GPU instance | $650 |
| Cloud storage for the prompt dataset and artifacts | ~200-500 GB | $60 |
| Printing and presentation materials | Flat rate | $50 |
| Miscellaneous software and API costs | Estimated | $240 |
| Total |  | $1,000 |

## Expected Outcomes and Impact

By the end of the 10-week program, I expect to produce:

- a curated dataset of LLM-assisted development prompts with weak labels and metadata
- a validated prompt taxonomy learned from embeddings and manual review
- baseline ML models for prompt classification and representation analysis
- one optional PEFT experiment if the data volume supports it
- a final report and presentation suitable for a workshop or conference submission

More broadly, the project contributes to the study of how pretrained language representations can make sense of real-world developer prompts under limited supervision. The findings should be useful both for researchers interested in prompt analysis and for tool builders designing better AI-assisted coding systems.

## Qualifications and Preparation

I am a junior Computer Science major at UCI with a 3.78 GPA and a strong background in the machine learning methods this project requires. My relevant preparation includes:

- coursework in supervised and unsupervised learning, including clustering and representation learning
- hands-on data analysis projects using real datasets and predictive modeling
- familiarity with Python data science libraries such as pandas, scikit-learn, and Hugging Face Transformers
- independent study of NLP fundamentals and repository mining techniques

I am comfortable working independently on open-ended research problems and motivated by research that connects ML methods to real-world human behavior.

## Responsibility

This project is investigator-initiated and independently conducted. My faculty mentor is Dr. Thomas Zimmermann, whose prior work on AI-assisted development informs the broader framing of the research. I will meet with Dr. Zimmermann one to two times per week to discuss progress, troubleshoot methodological challenges, and receive feedback on analysis and writing.

The day-to-day research work is my own. I am responsible for dataset construction, prompt annotation, representation learning experiments, clustering, lightweight predictive modeling, any optional PEFT experiment, and writing all deliverables. Dr. Zimmermann will serve in an advisory capacity, reviewing findings and helping interpret them in the context of the broader literature.

I will maintain a shared progress log and send weekly written updates summarizing completed tasks, blockers, and next steps. This structure ensures steady progress while preserving the independence of the research.

## Limitations

This project is intentionally designed for limited data. That means:

- full fine-tuning of a large generative model is not the primary objective
- some outcome signals may be missing or noisy
- the strongest claims will be about representation quality and prompt taxonomy, not causal impact

These are not weaknesses of the proposal so much as realistic boundaries that align the methods with the available evidence.

## References

[1] Vaithilingam, P., Zhang, T., & Glassman, E. L. (2022). Expectation vs. Experience: Evaluating the Usability of Code Generation Tools Powered by Large Language Models. In CHI Conference on Human Factors in Computing Systems Extended Abstracts (CHI EA '22). ACM. https://doi.org/10.1145/3491101.3519665  
[2] Barke, S., James, M. B., & Polikarpova, N. (2023). Grounded Copilot: How Programmers Interact with Code-Generating Models. Proceedings of the ACM on Programming Languages, 7(OOPSLA1), 85-111. https://doi.org/10.1145/3586030  
[3] Ziegler, A., Kalliamvakou, E., Li, X. A., Rice, A., Rifkin, D., Simister, G., & Aftandilian, E. (2022). Productivity Assessment of Neural Code Completion. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming (MAPS '22), 21-29. https://doi.org/10.1145/3520312.3534864  
[4] Bird, C., Ford, D., Zimmermann, T., Forsgren, N., Kalliamvakou, E., Lowdermilk, T., & Gazit, I. (2023). Taking Flight with Copilot: Early Insights and Opportunities of AI-Powered Pair-Programming Tools. ACM Queue, 20(6), 35-57. https://doi.org/10.1145/3582083  
[5] Wang, R., Cheng, R., Ford, D., & Zimmermann, T. (2024). Investigating and Designing for Trust in AI-Powered Code Generation Tools. In Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24), 1475-1493. https://doi.org/10.1145/3630106.3658984  
[6] Dyer, R., Nguyen, H. A., Rajan, H., & Nguyen, T. N. (2013). Boa: A Language and Infrastructure for Analyzing Ultra-Large-Scale Software Repositories. In Proceedings of the 35th International Conference on Software Engineering (ICSE '13), 422-431. https://doi.org/10.1109/ICSE.2013.6606588  
[7] GitHub REST API Documentation. https://docs.github.com/en/rest  
[8] GHTorrent Project. https://ghtorrent.org
