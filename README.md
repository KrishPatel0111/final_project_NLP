# Measuring Cultural and Contextual Cue Loss in LLM Summaries

Systematic evaluation of how LLMs preserve cultural and contextual cues in summaries.

**Authors:** Anooj Ravichandran, Krish Patel, Aarini Shaah

---

## 📋 Quick Setup

### 1. Clone Repository
```bash
git clone https://github.com/KrishPatel0111/final_project_NLP.git
cd final_project_NLP
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Analysis
```bash
python analysis/analysis.py
```
Results will be saved to `data/analysis/`

---

## 📊 What You'll Get

The script generates **14 CSV files** and **3 plots**:

### CSV Files
- `1_preservation_by_model.csv` - Performance by model
- `2_preservation_by_model_and_domain.csv` - Model × domain breakdown
- `3a_preservation_by_cue_type_per_model.csv` - Cultural vs Contextual per model
- `3b_preservation_by_cue_type_overall.csv` - Overall cue type comparison
- `4_preservation_by_subtype_top20.csv` - Top 20 cue subtypes
- `5_preservation_by_format.csv` - Summary format comparison
- `6_compression_*.csv` - Compression ratio analysis (2 files)
- `7a_pairwise_ttests.csv` - Statistical comparisons between models
- `7b_statistical_tests_summary.csv` - ANOVA and Chi-Square tests
- `9_preservation_by_cue_density.csv` - Article complexity effects
- `10a_best_preserved.csv` - Top 10 best preserved summaries
- `10b_worst_preserved.csv` - Top 10 worst preserved summaries

### Plots (in `data/analysis/plots/`)
- `8a_boxplot_by_model.png` - Distribution by model
- `8b_histogram_overall.png` - Overall preservation distribution
- `8c_cultural_vs_contextual_by_model.png` - Cue type comparison

- Aarini Shaah - aarini@example.com

**Project:** https://github.com/your-username/llm-cue-preservation
