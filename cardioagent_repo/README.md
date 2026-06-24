# CardioAgent

Temporal Rhythm-Burden Modeling and Guideline-Grounded Clinical Decision
Support for Wearable ECG Monitoring.

A four-tier pipeline for continuous wearable ECG monitoring:
- **Tier 1** — Signal Quality Agent (1D CNN)
- **Tier 2** — Arrhythmia Classifier (12-lead Transformer, lead-dropout augmented)
- **Tier 3** — Temporal Aggregator (BiLSTM + attention pooling)
- **Tier 4** — Clinical Triage Agent (RAG + LLM hybrid)

## Repository Structure

```
cardioagent/
├── scripts/          # All training and evaluation scripts
├── figures/          # Scripts to regenerate paper figures
└── README.md
```

## Datasets

All datasets are publicly available on PhysioNet, no credentialed access required:
- [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/)
- [MIT-BIH Noise Stress Test Database](https://physionet.org/content/nstdb/1.0.0/)
- [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- [Long-Term AF Database (LTAF)](https://physionet.org/content/ltafdb/1.0.0/)
- [Chapman-Shaoxing ECG Database](https://physionet.org/content/ecg-arrhythmia/1.0.0/)

## Setup

```bash
bash scripts/setup.sh
bash scripts/download_data.sh      # downloads all datasets
# or, for targeted downloads only:
bash scripts/download_ltaf_only.sh
bash scripts/download_nstdb_only.sh
```

## Training

```bash
python scripts/tier1_train.py
python scripts/tier2_12lead.py      # lead-dropout augmented, ~3 hrs on RTX 4090
python scripts/tier3_train.py       # attention pooling, ~1.5 hrs
python scripts/tier4_triage.py
```

## Evaluation

```bash
python scripts/cross_dataset_eval.py
python scripts/latency_test.py
python scripts/attention_viz.py
python scripts/ablation_study.py    # real-model, real-noise severity sweep on PTB-XL
```

## Key Results

| Component | Metric | Result |
|---|---|---|
| Tier 1 | Macro F1 | 0.819 |
| Tier 2 (12-lead) | AUROC | 0.876 |
| Tier 2 (single-lead) | AUROC | 0.771 |
| Tier 2 (cross-dataset, Chapman) | AUROC | 0.800 |
| Tier 3 | Mean MAE (all 5 dims pass target) | 0.0134 |
| Tier 4 | Guideline compliance | 100% (n=100) |

See the paper (`paper/main.tex`) for full methodology, the noise-severity
ablation study, and discussion of limitations.

## Important Note on LTAF Compatibility

The LTAF Database uses non-standard ambulatory electrode channel labelling
(generic "ECG" channel names rather than standard 12-lead nomenclature).
Direct Tier-2 inference on LTAF signal is therefore not meaningful; LTAF
is used only for Tier-3 training, where ground-truth rhythm labels (not
raw signal compatibility with the Tier-2 classifier) are what matters.
The noise-robustness ablation study evaluates Tier 2 on PTB-XL instead,
using real noise from MIT-BIH NSTDB. See the paper's Experimental
Methodology section for full details.


## Citation

```bibtex
@article{cardioagent2026,
  title={CardioAgent: Temporal Rhythm-Burden Modeling and Guideline-Grounded
         Clinical Decision Support for Wearable ECG Monitoring},
  author={Pandey, Sarthak and Abbas, Zeeshan and Rehman, Mobeen Ur},
  journal={Internet of Things},
  year={2026}
}
```

## License

[Add license — MIT or Apache 2.0 recommended for research code]

## Data Availability

All datasets used are publicly available on PhysioNet without credentialed
access. Code and trained weights are released in this repository.
