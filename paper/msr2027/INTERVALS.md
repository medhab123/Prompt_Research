# Intervals for the label-validity numbers

Cluster bootstrap over repositories, 2,000 resamples.

## Rule labeler vs LLM judge, whole corpus

- agreement: 0.521 [0.501, 0.542] (n=8404, 251 repositories)
- Cohen's kappa: 0.349 [0.325, 0.372]
- implement: both/either 0.470 [0.444, 0.497]
- debug: both/either 0.353 [0.319, 0.388]
- explain: both/either 0.218 [0.192, 0.242]
- review: both/either 0.069 [0.044, 0.098]
- tool_only: both/either 0.224 [0.187, 0.262]
- other: both/either 0.399 [0.333, 0.465]

## Against the human reference set (n=98)

- LLM judge (production definitions): accuracy 0.867 [0.794, 0.936]; kappa 0.814 [0.712, 0.910]
- LLM judge (prose definitions): accuracy 0.857 [0.778, 0.928]; kappa 0.801 [0.693, 0.894]
- LLM judge (earlier definitions): accuracy 0.847 [0.765, 0.918]; kappa 0.786 [0.671, 0.890]
- rule labeler: accuracy 0.459 [0.361, 0.551]; kappa 0.251 [0.134, 0.364]
- judge vs rule labeler, exact McNemar: judge only correct 43, rule only correct 3, p=4.62e-10

Per-category against the human reference set (precision / recall / n):

- judge: implement 0.94/0.89 (n=35), debug 0.92/0.89 (n=37), explain 0.80/0.92 (n=13), review 1.00/0.75 (n=8), tool_only 0.50/0.67 (n=3), other 0.25/0.50 (n=2)
- rule: implement 0.51/0.71 (n=35), debug 0.72/0.35 (n=37), explain 0.37/0.54 (n=13), review 0.00/0.00 (n=8), tool_only 0.00/0.00 (n=3), other 0.00/0.00 (n=2)

## One definition text vs another, same model and inputs

- labels changed: 0.158 [0.148, 0.170] of 8398 turns
- kappa between the two label sets: 0.788 [0.775, 0.802]
