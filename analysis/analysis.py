"""
Enhanced Cue Preservation Analysis
Includes statistical tests, visualizations, and advanced metrics
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Paths
PRESERVATION_FILE = 'data/extractions/preservation_results.jsonl'
SUMMARIES_FILE = 'data/outputs/all_summaries.jsonl'
OUTPUT_DIR = 'data/analysis'

# Create output directories
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(f'{OUTPUT_DIR}/plots').mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data():
    """Load preservation results and summaries."""
    print("📂 Loading data...")
    
    # Load preservation results
    results = []
    with open(PRESERVATION_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    
    # Load summaries (for length analysis)
    summaries = {}
    with open(SUMMARIES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            key = f"{data['title']}|{data['model']}|{data['style']}"
            summaries[key] = data
    
    # Add summary lengths to results
    for result in results:
        key = f"{result['title']}|{result['summary_model']}|{result['summary_format']}"
        if key in summaries:
            result['summary_length'] = len(summaries[key]['summary'].split())
            result['summary_text'] = summaries[key]['summary']
        else:
            result['summary_length'] = 0
            result['summary_text'] = ""
    
    print(f"✅ Loaded {len(results)} preservation results")
    return results


def analyze_by_model(results):
    """Analyze preservation rates by model."""
    print("\n" + "="*80)
    print("📊 1. PRESERVATION BY MODEL")
    print("="*80)
    
    model_stats = defaultdict(lambda: {
        'count': 0,
        'total_cues': 0,
        'preserved_cues': 0,
        'preservation_rates': []
    })
    
    for result in results:
        model = result['summary_model']
        model_stats[model]['count'] += 1
        model_stats[model]['total_cues'] += result['article_total_cues']
        model_stats[model]['preserved_cues'] += result['cues_preserved']
        model_stats[model]['preservation_rates'].append(result['preservation_rate'])
    
    # Calculate averages
    summary_data = []
    for model, stats in sorted(model_stats.items()):
        avg_rate = np.mean(stats['preservation_rates'])
        std_rate = np.std(stats['preservation_rates'])
        overall_rate = stats['preserved_cues'] / stats['total_cues'] if stats['total_cues'] > 0 else 0
        
        summary_data.append({
            'Model': model,
            'Summaries': stats['count'],
            'Total Cues': stats['total_cues'],
            'Preserved Cues': stats['preserved_cues'],
            'Avg Preservation Rate': f"{avg_rate:.2%}",
            'Std Dev': f"{std_rate:.2%}",
            'Overall Preservation Rate': f"{overall_rate:.2%}"
        })
        
        print(f"\n{model}:")
        print(f"  Summaries: {stats['count']}")
        print(f"  Total cues: {stats['total_cues']:,}")
        print(f"  Preserved: {stats['preserved_cues']:,} ({overall_rate:.2%})")
        print(f"  Avg rate: {avg_rate:.2%} ± {std_rate:.2%}")
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{OUTPUT_DIR}/1_preservation_by_model.csv', index=False)
    print(f"\n✅ Saved to {OUTPUT_DIR}/1_preservation_by_model.csv")
    
    return model_stats


def analyze_by_model_and_domain(results):
    """Analyze preservation rates by model and domain."""
    print("\n" + "="*80)
    print("📊 2. PRESERVATION BY MODEL AND DOMAIN")
    print("="*80)
    
    model_domain_stats = defaultdict(lambda: defaultdict(lambda: {
        'count': 0,
        'total_cues': 0,
        'preserved_cues': 0,
        'preservation_rates': []
    }))
    
    for result in results:
        model = result['summary_model']
        domain = result['domain']
        
        stats = model_domain_stats[model][domain]
        stats['count'] += 1
        stats['total_cues'] += result['article_total_cues']
        stats['preserved_cues'] += result['cues_preserved']
        stats['preservation_rates'].append(result['preservation_rate'])
    
    summary_data = []
    for model in sorted(model_domain_stats.keys()):
        print(f"\n{model}:")
        for domain in sorted(model_domain_stats[model].keys()):
            stats = model_domain_stats[model][domain]
            avg_rate = np.mean(stats['preservation_rates'])
            overall_rate = stats['preserved_cues'] / stats['total_cues'] if stats['total_cues'] > 0 else 0
            
            summary_data.append({
                'Model': model,
                'Domain': domain,
                'Summaries': stats['count'],
                'Total Cues': stats['total_cues'],
                'Preserved Cues': stats['preserved_cues'],
                'Avg Preservation Rate': f"{avg_rate:.2%}",
                'Overall Preservation Rate': f"{overall_rate:.2%}"
            })
            
            print(f"  {domain}: {stats['count']} summaries, {overall_rate:.2%}")
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{OUTPUT_DIR}/2_preservation_by_model_and_domain.csv', index=False)
    print(f"\n✅ Saved to {OUTPUT_DIR}/2_preservation_by_model_and_domain.csv")
    
    return model_domain_stats


def analyze_by_cue_type(results):
    """Analyze by cue type (Cultural vs Contextual)."""
    print("\n" + "="*80)
    print("📊 3. PRESERVATION BY CUE TYPE")
    print("="*80)
    
    # Per model
    model_cue_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'preserved': 0}))
    
    for result in results:
        model = result['summary_model']
        for detail in result['preservation_details']:
            cue_type = detail['cue_type']
            model_cue_stats[model][cue_type]['total'] += 1
            if detail['preserved']:
                model_cue_stats[model][cue_type]['preserved'] += 1
    
    # Per model summary
    print("\nPer Model:")
    summary_data_model = []
    for model in sorted(model_cue_stats.keys()):
        print(f"\n{model}:")
        for cue_type in sorted(model_cue_stats[model].keys()):
            stats = model_cue_stats[model][cue_type]
            rate = stats['preserved'] / stats['total'] if stats['total'] > 0 else 0
            
            summary_data_model.append({
                'Model': model,
                'Cue Type': cue_type,
                'Total Cues': stats['total'],
                'Preserved Cues': stats['preserved'],
                'Preservation Rate': f"{rate:.2%}"
            })
            
            print(f"  {cue_type}: {stats['preserved']:,}/{stats['total']:,} ({rate:.2%})")
    
    df = pd.DataFrame(summary_data_model)
    df.to_csv(f'{OUTPUT_DIR}/3a_preservation_by_cue_type_per_model.csv', index=False)
    
    # Overall
    print("\nOverall (across all models):")
    cue_type_stats = defaultdict(lambda: {'total': 0, 'preserved': 0})
    
    for result in results:
        for detail in result['preservation_details']:
            cue_type = detail['cue_type']
            cue_type_stats[cue_type]['total'] += 1
            if detail['preserved']:
                cue_type_stats[cue_type]['preserved'] += 1
    
    summary_data_overall = []
    for cue_type in sorted(cue_type_stats.keys()):
        stats = cue_type_stats[cue_type]
        rate = stats['preserved'] / stats['total'] if stats['total'] > 0 else 0
        
        summary_data_overall.append({
            'Cue Type': cue_type,
            'Total Cues': stats['total'],
            'Preserved Cues': stats['preserved'],
            'Preservation Rate': f"{rate:.2%}"
        })
        
        print(f"\n{cue_type}:")
        print(f"  Total: {stats['total']:,}")
        print(f"  Preserved: {stats['preserved']:,} ({rate:.2%})")
    
    df = pd.DataFrame(summary_data_overall)
    df.to_csv(f'{OUTPUT_DIR}/3b_preservation_by_cue_type_overall.csv', index=False)
    
    print(f"\n✅ Saved to {OUTPUT_DIR}/3a_* and 3b_*")
    
    return model_cue_stats, cue_type_stats


def analyze_by_subtype(results):
    """Analyze by cue subtype."""
    print("\n" + "="*80)
    print("📊 4. PRESERVATION BY CUE SUBTYPE (TOP 20)")
    print("="*80)
    
    subtype_stats = defaultdict(lambda: {
        'total': 0,
        'preserved': 0,
        'cue_type': None
    })
    
    for result in results:
        for detail in result['preservation_details']:
            subtype = detail['subtype']
            subtype_stats[subtype]['total'] += 1
            subtype_stats[subtype]['cue_type'] = detail['cue_type']
            if detail['preserved']:
                subtype_stats[subtype]['preserved'] += 1
    
    # Sort by total
    sorted_subtypes = sorted(subtype_stats.items(), key=lambda x: x[1]['total'], reverse=True)
    
    summary_data = []
    for subtype, stats in sorted_subtypes[:20]:
        rate = stats['preserved'] / stats['total'] if stats['total'] > 0 else 0
        summary_data.append({
            'Subtype': subtype,
            'Cue Type': stats['cue_type'],
            'Total': stats['total'],
            'Preserved': stats['preserved'],
            'Rate': f"{rate:.2%}"
        })
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{OUTPUT_DIR}/4_preservation_by_subtype_top20.csv', index=False)
    print(f"\n{df.to_string(index=False)}")
    print(f"\n✅ Saved to {OUTPUT_DIR}/4_preservation_by_subtype_top20.csv")


def analyze_by_format(results):
    """Analyze by summary format."""
    print("\n" + "="*80)
    print("📊 5. PRESERVATION BY SUMMARY FORMAT")
    print("="*80)
    
    format_stats = defaultdict(lambda: {
        'count': 0,
        'total_cues': 0,
        'preserved_cues': 0,
        'preservation_rates': []
    })
    
    for result in results:
        fmt = result['summary_format']
        format_stats[fmt]['count'] += 1
        format_stats[fmt]['total_cues'] += result['article_total_cues']
        format_stats[fmt]['preserved_cues'] += result['cues_preserved']
        format_stats[fmt]['preservation_rates'].append(result['preservation_rate'])
    
    summary_data = []
    for fmt in sorted(format_stats.keys()):
        stats = format_stats[fmt]
        avg_rate = np.mean(stats['preservation_rates'])
        overall_rate = stats['preserved_cues'] / stats['total_cues'] if stats['total_cues'] > 0 else 0
        
        summary_data.append({
            'Format': fmt,
            'Summaries': stats['count'],
            'Avg Preservation': f"{avg_rate:.2%}",
            'Overall Preservation': f"{overall_rate:.2%}"
        })
        
        print(f"\n{fmt}: {stats['count']} summaries, {overall_rate:.2%}")
    
    df = pd.DataFrame(summary_data)
    df.to_csv(f'{OUTPUT_DIR}/5_preservation_by_format.csv', index=False)
    print(f"\n✅ Saved to {OUTPUT_DIR}/5_preservation_by_format.csv")


def analyze_compression_ratio(results):
    """Analyze preservation vs compression ratio."""
    print("\n" + "="*80)
    print("📊 6. COMPRESSION RATIO ANALYSIS")
    print("="*80)
    
    # Calculate compression ratios
    data = []
    for result in results:
        if result['summary_length'] > 0:
            # Estimate article length from cue density (rough approximation)
            # Average: 1 cue per ~50 words
            estimated_article_length = result['article_total_cues'] * 50
            compression_ratio = result['summary_length'] / estimated_article_length
            
            data.append({
                'model': result['summary_model'],
                'domain': result['domain'],
                'format': result['summary_format'],
                'summary_length': result['summary_length'],
                'article_cues': result['article_total_cues'],
                'compression_ratio': compression_ratio,
                'preservation_rate': result['preservation_rate']
            })
    
    df = pd.DataFrame(data)
    
    # Correlation analysis
    corr = df[['compression_ratio', 'preservation_rate']].corr()
    print(f"\nCorrelation between compression ratio and preservation rate:")
    print(f"  r = {corr.iloc[0, 1]:.3f}")
    
    # By model
    print("\nAverage compression ratio by model:")
    model_compression = df.groupby('model').agg({
        'compression_ratio': 'mean',
        'preservation_rate': 'mean',
        'summary_length': 'mean'
    }).round(4)
    print(model_compression)
    
    model_compression.to_csv(f'{OUTPUT_DIR}/6_compression_by_model.csv')
    df.to_csv(f'{OUTPUT_DIR}/6_compression_detailed.csv', index=False)
    print(f"\n✅ Saved to {OUTPUT_DIR}/6_compression_*.csv")
    
    return df


def statistical_significance_tests(results, model_stats, cue_type_stats):
    """Perform statistical tests."""
    print("\n" + "="*80)
    print("📊 7. STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)
    
    # Prepare data for tests
    models = list(model_stats.keys())
    
    # 1. ANOVA: Are model differences significant?
    print("\n1. ANOVA: Model Differences")
    model_groups = [model_stats[model]['preservation_rates'] for model in models]
    f_stat, p_value = sp_stats.f_oneway(*model_groups)
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   p-value: {p_value:.4f}")
    print(f"   Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)")
    
    # 2. Pairwise t-tests between models
    print("\n2. Pairwise T-Tests Between Models:")
    pairwise_results = []
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            rates1 = model_stats[model1]['preservation_rates']
            rates2 = model_stats[model2]['preservation_rates']
            t_stat, p_val = sp_stats.ttest_ind(rates1, rates2)
            
            pairwise_results.append({
                'Model 1': model1,
                'Model 2': model2,
                'T-statistic': f"{t_stat:.4f}",
                'P-value': f"{p_val:.4f}",
                'Significant': 'Yes' if p_val < 0.05 else 'No'
            })
            
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"   {model1} vs {model2}: t={t_stat:.3f}, p={p_val:.4f} {sig}")
    
    df = pd.DataFrame(pairwise_results)
    df.to_csv(f'{OUTPUT_DIR}/7a_pairwise_ttests.csv', index=False)
    
    # 3. Chi-square: Cultural vs Contextual preservation
    print("\n3. Chi-Square: Cultural vs Contextual Cues")
    contingency_table = []
    for cue_type, stats in cue_type_stats.items():
        contingency_table.append([stats['preserved'], stats['total'] - stats['preserved']])
    
    chi2, p_val, dof, expected = chi2_contingency(contingency_table)
    print(f"   χ² = {chi2:.4f}, df = {dof}, p = {p_val:.4f}")
    print(f"   Result: {'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'} difference")
    
    # Save summary
    test_summary = {
        'Test': ['ANOVA (Models)', 'Chi-Square (Cue Types)'],
        'Statistic': [f"{f_stat:.4f}", f"{chi2:.4f}"],
        'P-value': [f"{p_value:.4f}", f"{p_val:.4f}"],
        'Significant (α=0.05)': ['Yes' if p_value < 0.05 else 'No', 'Yes' if p_val < 0.05 else 'No']
    }
    
    df = pd.DataFrame(test_summary)
    df.to_csv(f'{OUTPUT_DIR}/7b_statistical_tests_summary.csv', index=False)
    print(f"\n✅ Saved to {OUTPUT_DIR}/7a_* and 7b_*")


def create_visualizations(results, model_stats):
    """Create distribution plots."""
    print("\n" + "="*80)
    print("📊 8. CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Boxplot: Preservation by model
    print("\nGenerating boxplot...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    plot_data = []
    for result in results:
        plot_data.append({
            'Model': result['summary_model'],
            'Preservation Rate': result['preservation_rate']
        })
    
    df = pd.DataFrame(plot_data)
    sns.boxplot(data=df, x='Model', y='Preservation Rate', ax=ax)
    ax.set_title('Preservation Rate Distribution by Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Preservation Rate', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/8a_boxplot_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Histogram: Overall distribution
    print("Generating histogram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    preservation_rates = [r['preservation_rate'] for r in results]
    ax.hist(preservation_rates, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(preservation_rates), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(preservation_rates):.2%}')
    ax.set_title('Distribution of Preservation Rates (All Summaries)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Preservation Rate', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/8b_histogram_overall.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Bar chart: Cultural vs Contextual by model
    print("Generating bar chart...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bar_data = []
    for result in results:
        cultural_count = sum(1 for d in result['preservation_details'] if d['cue_type'] == 'Cultural')
        contextual_count = sum(1 for d in result['preservation_details'] if d['cue_type'] == 'Contextual')
        cultural_preserved = sum(1 for d in result['preservation_details'] if d['cue_type'] == 'Cultural' and d['preserved'])
        contextual_preserved = sum(1 for d in result['preservation_details'] if d['cue_type'] == 'Contextual' and d['preserved'])
        
        cultural_rate = cultural_preserved / cultural_count if cultural_count > 0 else 0
        contextual_rate = contextual_preserved / contextual_count if contextual_count > 0 else 0
        
        bar_data.append({
            'Model': result['summary_model'],
            'Cultural': cultural_rate,
            'Contextual': contextual_rate
        })
    
    df = pd.DataFrame(bar_data)
    df_grouped = df.groupby('Model').mean()
    
    df_grouped.plot(kind='bar', ax=ax)
    ax.set_title('Cultural vs Contextual Cue Preservation by Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Preservation Rate', fontsize=12)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(title='Cue Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/plots/8c_cultural_vs_contextual_by_model.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Saved 3 plots to {OUTPUT_DIR}/plots/")


def analyze_article_characteristics(results):
    """Analyze how article features affect preservation."""
    print("\n" + "="*80)
    print("📊 9. ARTICLE CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    # Cue density analysis
    data = []
    for result in results:
        cue_density = result['article_total_cues']  # Absolute count
        data.append({
            'article': result['title'][:50],
            'domain': result['domain'],
            'model': result['summary_model'],
            'total_cues': result['article_total_cues'],
            'preservation_rate': result['preservation_rate']
        })
    
    df = pd.DataFrame(data)
    
    # Correlation: cues vs preservation
    corr = df[['total_cues', 'preservation_rate']].corr()
    print(f"\nCorrelation between total cues and preservation:")
    print(f"  r = {corr.iloc[0, 1]:.3f}")
    
    # Bucketing by cue count
    df['cue_bucket'] = pd.cut(df['total_cues'], bins=[0, 50, 100, 150, 300], labels=['<50', '50-100', '100-150', '150+'])
    
    print("\nPreservation by cue count bucket:")
    bucket_stats = df.groupby('cue_bucket')['preservation_rate'].agg(['count', 'mean', 'std']).round(4)
    print(bucket_stats)
    
    bucket_stats.to_csv(f'{OUTPUT_DIR}/9_preservation_by_cue_density.csv')
    print(f"\n✅ Saved to {OUTPUT_DIR}/9_preservation_by_cue_density.csv")


def find_extreme_cases(results):
    """Find best and worst preservation examples."""
    print("\n" + "="*80)
    print("📊 10. EXTREME CASES (BEST/WORST)")
    print("="*80)
    
    # Sort by preservation rate
    sorted_results = sorted(results, key=lambda x: x['preservation_rate'], reverse=True)
    
    # Top 10 best
    print("\nTop 10 Best Preserved:")
    best_cases = []
    for i, result in enumerate(sorted_results[:10], 1):
        best_cases.append({
            'Rank': i,
            'Title': result['title'][:80],
            'Model': result['summary_model'],
            'Domain': result['domain'],
            'Preservation Rate': f"{result['preservation_rate']:.2%}",
            'Total Cues': result['article_total_cues'],
            'Preserved': result['cues_preserved']
        })
        print(f"{i}. {result['title'][:60]}... ({result['summary_model']}, {result['preservation_rate']:.2%})")
    
    df = pd.DataFrame(best_cases)
    df.to_csv(f'{OUTPUT_DIR}/10a_best_preserved.csv', index=False)
    
    # Bottom 10 worst
    print("\nTop 10 Worst Preserved:")
    worst_cases = []
    for i, result in enumerate(sorted_results[-10:], 1):
        worst_cases.append({
            'Rank': i,
            'Title': result['title'][:80],
            'Model': result['summary_model'],
            'Domain': result['domain'],
            'Preservation Rate': f"{result['preservation_rate']:.2%}",
            'Total Cues': result['article_total_cues'],
            'Preserved': result['cues_preserved']
        })
        print(f"{i}. {result['title'][:60]}... ({result['summary_model']}, {result['preservation_rate']:.2%})")
    
    df = pd.DataFrame(worst_cases)
    df.to_csv(f'{OUTPUT_DIR}/10b_worst_preserved.csv', index=False)
    
    print(f"\n✅ Saved to {OUTPUT_DIR}/10a_* and 10b_*")


def main():
    """Run all analyses."""
    
    print("="*80)
    print("ENHANCED CUE PRESERVATION ANALYSIS")
    print("="*80)
    
    # Load data
    results = load_data()
    
    # Run analyses
    model_stats = analyze_by_model(results)
    model_domain_stats = analyze_by_model_and_domain(results)
    model_cue_stats, cue_type_stats = analyze_by_cue_type(results)
    analyze_by_subtype(results)
    analyze_by_format(results)
    
    # NEW ANALYSES
    compression_df = analyze_compression_ratio(results)
    statistical_significance_tests(results, model_stats, cue_type_stats)
    create_visualizations(results, model_stats)
    analyze_article_characteristics(results)
    find_extreme_cases(results)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
    print("\nGenerated files:")
    print("  1. preservation_by_model.csv")
    print("  2. preservation_by_model_and_domain.csv")
    print("  3a. preservation_by_cue_type_per_model.csv")
    print("  3b. preservation_by_cue_type_overall.csv")
    print("  4. preservation_by_subtype_top20.csv")
    print("  5. preservation_by_format.csv")
    print("  6. compression_* (2 files)")
    print("  7. statistical_tests_* (2 files)")
    print("  8. plots/ (3 visualizations)")
    print("  9. preservation_by_cue_density.csv")
    print("  10. best_preserved.csv & worst_preserved.csv")
    print(f"\n📊 Total: 14+ files generated")


if __name__ == "__main__":
    main()