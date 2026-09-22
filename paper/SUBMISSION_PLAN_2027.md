# MSR 2027 and FORGE 2027: submission plan

Checked against the official calls for papers on 2026-09-21.

| | MSR 2027 Technical Papers | FORGE 2027 Research Papers |
|---|---|---|
| Deadlines (AoE) | abstract **20 Oct**, paper **23 Oct 2026** | abstract **25 Oct**, paper **30 Oct 2026** |
| Length | full: 10 pp + 2 pp refs; short/WIP: 4 + 1 | full: 10 pp (everything) + 2 pp refs; new idea: 4 + 2 |
| Template | IEEE conference proceedings | IEEE conference proceedings |
| Review | double-anonymous | double-anonymous |
| Criteria | soundness, relevance, novelty, presentation, **replicability** | novelty, rigor, relevance, **verifiability & transparency**, presentation |
| Open science | **expected**: code + data + instructions, archived (Zenodo/Figshare/OSF), open license | strongly encouraged |
| AI use | must be disclosed in the acknowledgments beyond copy-editing; unverified AI text → desk reject | (check the full CFP) |

## The concurrent-submission rule decides the split

MSR states that submitted papers "must not be under review or submitted for review elsewhere." FORGE (IEEE/ACM) has the same rule. The two papers therefore have to be **different papers**, not two framings of one:

| | MSR: label-validity paper | FORGE: prediction paper |
|---|---|---|
| Question | Are mined agent-action labels trustworthy, and how much do conclusions depend on how they are built? | Can an agent's next action be predicted before it responds, and does that depend on the label? |
| Owns | dataset + pipeline, rule/LLM/human agreement, disagreement taxonomy C0–C7, prevalence shifts, definition-wording flip, parser-defect case | all prediction models, history/prev-action results, label-source effect on prediction |
| Must not contain | any prediction result | the disagreement taxonomy, the prevalence analysis, the parser-defect analysis |

Other points to settle:
- The corpus description will overlap. Keep it short in FORGE and cite the MSR paper anonymously as a companion study. Ask your advisor, and if unsure the FORGE PC chairs, whether this overlap is acceptable.
- The IAB @ NeurIPS workshop version was non-archival, which generally does not count as prior publication. Still:
  - confirm FORGE's policy
  - don't reuse the IAB title, "One Turn Back"
- arXiv preprints must use a different title and must not name the venue (MSR rule).

## Drafts

- `msr2027/intro_and_progress.tex`
- `forge2027/intro_and_progress.tex`
- `references_2027.bib`: verified entries

In each draft, the "Study Status" section is internal. Turn it into the method and results sections before submitting. Every number is traced in the `NUMBERS USED` comment at the end of each file.

## Corrections to earlier drafts

- **Related work.** `intro_msr_forge_draft.tex` and `main.tex` claim prior work reports no agreement between labelings. That is false:
  - Tang et al. report human κ = 0.669 and an LLM macro-F1 of 0.802 against humans.
  - Fang et al. report an LLM macro-F1 of 0.83.

  The real distinctions are that (a) they label *developer intent*, not *agent action*, and (b) each validates one construction, where we show that conclusions shift across constructions.
- **Scale.** Their corpora are about 5× larger than ours (1,300+ repos vs 251). Do not claim scale as a contribution.
- **Parser story.** The old intro says the parser bug inflated the history effect from +0.023 to +0.175. That change also includes relabeling and the switch from a single split to CV. The parser fix alone moved it from about +0.175 to +0.125 (best model per representation, single split, rule labels).
- **Stale numbers.** Numbers from before the Sept 17 repair are stale:
  - rule vs LLM agreement: κ 0.223 → **0.398** now
  - rule debug share: 3.5% → **15.2%**
  - sessions: 3,191 → **3,197**

## Critical path (about 4.5 weeks)

| By | MSR | FORGE |
|---|---|---|
| 28 Sep | Decide the human reference-set design: random sample, 2 annotators, κ | Re-run prediction on the repaired corpus (rule v3 + LLM) |
| 5 Oct | Re-translate the 225 error-page prompts and re-label them; annotate the reference set | Bootstrap CIs, repeated grouped CV, zero-shot LLM baseline |
| 12 Oct | Full draft; method + results sections | Full draft |
| 20 Oct | **Abstract**; Zenodo package (anonymized) | — |
| 23 Oct | **Paper** | Revise after MSR is submitted |
| 25 / 30 Oct | — | **Abstract / Paper** |

**Biggest risk.** Without a random, independently double-annotated human sample, reviewers will read the κ = 0.49 figure as coming from a disagreement-selected set graded by one author who could see the machine labels. For MSR, whose headline claim is label validity, this is the item most likely to decide acceptance.
