# Speed Dating: Stated vs. Revealed Preferences and the Halo Effect

A linear-models investigation of how attractiveness ratings overwhelm other partner traits in actual yes/no decisions, even when participants claim otherwise. Built on the Columbia Speed Dating Experiment (Fisman, Iyengar, Kamenica & Simonson, 2002–2004).

## Research questions

1. **Stated vs. revealed preferences.** How large is the gap between the trait weights participants report *before* the event and the weights implied by their actual yes/no decisions?
2. **Causal dominance of attractiveness.** Once we control for the collinearity induced by the *halo effect* (perceived beauty bleeding into perceived sincerity, intelligence, fun, ambition) and for participant heterogeneity, does attractiveness remain the decisive factor — or is it a statistical artifact?

## Approach (five stages)

The analysis is framed as a "trial of the halo effect." Each stage tests a falsifiable hypothesis with a concrete metric.

| Stage | Hypothesis | Method | Notebook |
|-------|------------|--------|----------|
| 1. Witness statements | Stated `attr` weight is significantly *below* the revealed weight | EDA, correlation heatmap, baseline LPM | [`notebooks/02_stage1_eda_baseline.ipynb`](notebooks/02_stage1_eda_baseline.ipynb) |
| 2. Cross-examination | In a 40+ feature set, regularization stably selects only `attr` and a few others | Elastic Net + Adaptive Lasso + Stability Selection (with `GroupKFold` on `iid`) | [`notebooks/03_stage2_regularization.ipynb`](notebooks/03_stage2_regularization.ipynb) |
| 3. Controlled courtroom | After participant fixed effects, `attr` remains the largest coefficient | `PanelOLS` with entity effects, clustered SEs | [`notebooks/04_stage3_fixed_effects.ipynb`](notebooks/04_stage3_fixed_effects.ipynb) |
| 4. Appeal | Every non-`attr` trait coefficient shrinks once `attr` enters — the halo's mathematical fingerprint | OVB decomposition for all five traits + Cinelli-Hazlett Sensemakr | [`notebooks/05_stage4_ovb_sensemakr.ipynb`](notebooks/05_stage4_ovb_sensemakr.ipynb) |
| 5. Verdict | Findings are stable across slices and model choices | Subsamples, leave-one-wave-out, logistic robustness | [`notebooks/06_stage5_robustness.ipynb`](notebooks/06_stage5_robustness.ipynb) |

The headline metric across all five stages is the **change in trait coefficients across specifications** — the shrinkage of the non-`attr` coefficients when `attr` is added is a direct measurement of the halo.

## Methodological notes

- **Linear Probability Model (LPM) is the main spec.** The course allows logistic regression; LPM is a deliberate choice because (i) participant fixed effects in Stage 3 add cleanly as entity dummies (logit needs conditional logit to dodge the incidental-parameters problem), (ii) Sensemakr in Stage 4 is implemented for OLS only, and (iii) LPM coefficients are directly interpretable as marginal probability changes — which matches the "weights" narrative. Logit appears in Stage 5 as a robustness check.
- **`GroupKFold` on `iid` is mandatory.** Each participant contributes 5–22 rows (median 16); naive `KFold` leaks information across folds and inflates CV scores. `ElasticNetCV` does not accept `groups`, so we use `GridSearchCV + ElasticNet`.
- **Waves 6–9 are dropped from the main analysis** because their stated-preference scale is different (1–10 ratings rather than a 100-point allocation). Kept for robustness in Stage 5.

## Repository layout

```
.
├── data/
│   ├── raw/           # Speed_Dating_Data.csv (195 cols, 8378 rows) + key
│   └── clean/         # cleaned.parquet (output of notebook 01)
├── notebooks/
│   ├── 00_data_exploration.ipynb
│   ├── 01_data_cleaning.ipynb
│   ├── 02_stage1_eda_baseline.ipynb
│   ├── 03_stage2_regularization.ipynb
│   ├── 04_stage3_fixed_effects.ipynb
│   ├── 05_stage4_ovb_sensemakr.ipynb
│   └── 06_stage5_robustness.ipynb
├── src/utils/         # Reusable Python helpers imported by the notebooks
├── figures/           # Plots produced by the notebooks
├── report/            # Final written report
├── plan/              # Internal planning docs (gitignored)
├── proposal/          # Submitted project proposal (gitignored)
└── reference/         # Reference papers (gitignored)
```

## Data

- **Source:** Fisman, Iyengar, Kamenica & Simonson (2006), *QJE*. Distributed via Kaggle as `annavictoria/speed-dating-experiment`.
- **File:** `data/raw/Speed_Dating_Data.csv` — 195 columns, 8378 rows, 551 participants across 21 waves. **Must** be loaded with `encoding='latin-1'`.
- **Do not use `speeddating.csv`** (the 123-column preprocessed version on Kaggle): it strips the `iid` / `pid` identifiers needed for participant fixed effects and clustered SEs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas numpy scipy scikit-learn statsmodels linearmodels \
            PySensemakr stability-selection matplotlib seaborn pyarrow jupyter
jupyter lab
```

Then run the notebooks in numerical order (`00 → 06`).

## Key references

- Fisman, Iyengar, Kamenica, Simonson (2006). *Gender Differences in Mate Selection: Evidence from a Speed Dating Experiment.* **QJE.**
- Cinelli & Hazlett (2020). *Making Sense of Sensitivity.* **JRSS-B.**
- Zou (2006). *The Adaptive Lasso and Its Oracle Properties.* **JASA.**
- Meinshausen & Bühlmann (2010). *Stability Selection.* **JRSS-B.**
- Eastwick & Finkel (2008). *Sex Differences in Mate Preferences Revisited.* **JPSP.**
