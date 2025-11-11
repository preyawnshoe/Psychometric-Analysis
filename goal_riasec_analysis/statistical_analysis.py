import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the combined RIASEC-Goal orientation dataset"""
    df = pd.read_csv('combined_riasec_goal_data.csv')
    print(f"Loaded data: {len(df)} participants")
    return df

def calculate_correlations(df):
    """Calculate comprehensive correlation analysis"""
    print("=== CORRELATION ANALYSIS ===")
    
    # Define dimension groups
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Calculate correlation matrix between all RIASEC and Goal dimensions
    correlation_results = {}
    significance_results = {}
    
    correlation_matrix = np.zeros((len(riasec_dims), len(goal_dims)))
    p_value_matrix = np.zeros((len(riasec_dims), len(goal_dims)))
    
    print("\nPearson Correlations between RIASEC and Goal Orientation Dimensions:")
    print("=" * 70)
    
    for i, riasec_dim in enumerate(riasec_dims):
        for j, goal_dim in enumerate(goal_dims):
            # Pearson correlation
            corr, p_value = pearsonr(df[riasec_dim], df[goal_dim])
            correlation_matrix[i, j] = corr
            p_value_matrix[i, j] = p_value
            
            # Store results
            key = f"{riasec_dim}_vs_{goal_dim}"
            correlation_results[key] = {
                'correlation': corr,
                'p_value': p_value,
                'significance': 'Significant' if p_value < 0.05 else 'Not Significant'
            }
            
            # Print results
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"{riasec_dim:<12} vs {goal_dim:<20}: r = {corr:6.3f}, p = {p_value:6.3f} {significance}")
    
    # Create correlation DataFrame for easy viewing
    correlation_df = pd.DataFrame(correlation_matrix, 
                                index=riasec_dims, 
                                columns=goal_dims)
    
    p_value_df = pd.DataFrame(p_value_matrix, 
                            index=riasec_dims, 
                            columns=goal_dims)
    
    # Save correlation results
    correlation_df.to_csv('riasec_goal_correlations.csv')
    p_value_df.to_csv('correlation_p_values.csv')
    
    # Find strongest correlations
    print("\n=== STRONGEST CORRELATIONS ===")
    strong_correlations = []
    for key, result in correlation_results.items():
        if abs(result['correlation']) > 0.3:  # Medium effect size
            strong_correlations.append((key, result['correlation'], result['p_value']))
    
    strong_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if strong_correlations:
        print("Correlations with |r| > 0.3:")
        for key, corr, p_val in strong_correlations:
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {key}: r = {corr:.3f}, p = {p_val:.3f} {significance}")
    else:
        print("No correlations with |r| > 0.3 found")
    
    return correlation_df, p_value_df, correlation_results

def calculate_comprehensive_statistics(df):
    """Calculate additional statistical measures"""
    print("\n=== COMPREHENSIVE STATISTICAL ANALYSIS ===")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Descriptive statistics
    print("\n1. DESCRIPTIVE STATISTICS")
    print("\nRIASEC Dimensions:")
    riasec_stats = df[riasec_dims].describe()
    print(riasec_stats.round(3))
    
    print("\nGoal Orientation Dimensions:")
    goal_stats = df[goal_dims].describe()
    print(goal_stats.round(3))
    
    # Normality tests
    print("\n2. NORMALITY TESTS (Shapiro-Wilk)")
    normality_results = {}
    
    print("\nRIASEC Dimensions:")
    for dim in riasec_dims:
        stat, p_value = stats.shapiro(df[dim])
        normality_results[f"RIASEC_{dim}"] = {'statistic': stat, 'p_value': p_value}
        normal = "Normal" if p_value > 0.05 else "Not Normal"
        print(f"  {dim}: W = {stat:.3f}, p = {p_value:.3f} ({normal})")
    
    print("\nGoal Orientation Dimensions:")
    for dim in goal_dims:
        stat, p_value = stats.shapiro(df[dim])
        normality_results[f"Goal_{dim}"] = {'statistic': stat, 'p_value': p_value}
        normal = "Normal" if p_value > 0.05 else "Not Normal"
        print(f"  {dim}: W = {stat:.3f}, p = {p_value:.3f} ({normal})")
    
    # Spearman correlations (non-parametric alternative)
    print("\n3. SPEARMAN RANK CORRELATIONS (Non-parametric)")
    spearman_results = {}
    
    for riasec_dim in riasec_dims:
        for goal_dim in goal_dims:
            rho, p_value = spearmanr(df[riasec_dim], df[goal_dim])
            key = f"{riasec_dim}_vs_{goal_dim}"
            spearman_results[key] = {'correlation': rho, 'p_value': p_value}
            
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            if abs(rho) > 0.3:  # Only print strong correlations
                print(f"  {riasec_dim} vs {goal_dim}: ρ = {rho:.3f}, p = {p_value:.3f} {significance}")
    
    # Individual participant analysis preview
    print("\n4. INDIVIDUAL PARTICIPANT PROFILES")
    print("\nTop 3 participants by Total RIASEC score:")
    top_riasec = df.nlargest(3, 'Total_RIASEC')[['participant', 'Total_RIASEC', 'Total_Goal_Orientation']]
    print(top_riasec.to_string(index=False))
    
    print("\nTop 3 participants by Total Goal Orientation score:")
    top_goal = df.nlargest(3, 'Total_Goal_Orientation')[['participant', 'Total_RIASEC', 'Total_Goal_Orientation']]
    print(top_goal.to_string(index=False))
    
    return normality_results, spearman_results

def identify_patterns(df, correlation_results):
    """Identify interesting patterns and insights"""
    print("\n=== PATTERN ANALYSIS ===")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Find dominant RIASEC type for each participant
    df['Dominant_RIASEC'] = df[riasec_dims].idxmax(axis=1)
    
    # Find dominant goal orientation for each participant
    df['Dominant_Goal'] = df[goal_dims].idxmax(axis=1)
    
    print("1. DOMINANT TYPES DISTRIBUTION")
    print("\nDominant RIASEC Types:")
    riasec_counts = df['Dominant_RIASEC'].value_counts()
    for riasec_type, count in riasec_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {riasec_type}: {count} participants ({percentage:.1f}%)")
    
    print("\nDominant Goal Orientations:")
    goal_counts = df['Dominant_Goal'].value_counts()
    for goal_type, count in goal_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {goal_type}: {count} participants ({percentage:.1f}%)")
    
    # Cross-tabulation of dominant types
    print("\n2. CROSS-TABULATION: Dominant RIASEC vs Dominant Goal")
    crosstab = pd.crosstab(df['Dominant_RIASEC'], df['Dominant_Goal'])
    print(crosstab)
    
    # Identify extreme profiles
    print("\n3. EXTREME PROFILES")
    
    # High achievers (high in both performance approach and mastery approach)
    high_achievers = df[(df['Performance_Approach'] > df['Performance_Approach'].quantile(0.75)) & 
                       (df['Mastery_Approach'] > df['Mastery_Approach'].quantile(0.75))]
    print(f"\nHigh Achievers (high Performance + Mastery Approach): {len(high_achievers)} participants")
    if len(high_achievers) > 0:
        print(high_achievers[['participant', 'Dominant_RIASEC', 'Performance_Approach', 'Mastery_Approach']].to_string(index=False))
    
    # Avoidance-oriented (high in both avoidance types)
    avoidance_oriented = df[(df['Performance_Avoidance'] > df['Performance_Avoidance'].quantile(0.75)) & 
                           (df['Mastery_Avoidance'] > df['Mastery_Avoidance'].quantile(0.75))]
    print(f"\nAvoidance-Oriented (high Performance + Mastery Avoidance): {len(avoidance_oriented)} participants")
    if len(avoidance_oriented) > 0:
        print(avoidance_oriented[['participant', 'Dominant_RIASEC', 'Performance_Avoidance', 'Mastery_Avoidance']].to_string(index=False))
    
    # Save updated dataset with dominant types
    df.to_csv('combined_riasec_goal_data_with_patterns.csv', index=False)
    
    return df

def generate_summary_report(correlation_results, normality_results, spearman_results):
    """Generate a comprehensive text summary"""
    print("\n=== GENERATING SUMMARY REPORT ===")
    
    report = []
    report.append("RIASEC-GOAL ORIENTATION CROSS-ANALYSIS SUMMARY REPORT")
    report.append("=" * 60)
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Sample Size: 14 participants")
    report.append("")
    
    # Methodology
    report.append("METHODOLOGY:")
    report.append("- RIASEC: Holland's Career Interest model (6 dimensions)")
    report.append("- Goal Orientation: 2x2 Achievement Goal Theory (4 dimensions)")
    report.append("- Statistical tests: Pearson correlation, Spearman correlation, Shapiro-Wilk normality")
    report.append("- Significance level: alpha = 0.05")
    report.append("")
    
    # Key findings
    report.append("KEY FINDINGS:")
    
    # Count significant correlations
    significant_pearson = sum(1 for result in correlation_results.values() if result['p_value'] < 0.05)
    report.append(f"- {significant_pearson} significant Pearson correlations found (p < 0.05)")
    
    # Strongest correlations
    strong_corrs = [(key, result['correlation']) for key, result in correlation_results.items() 
                   if abs(result['correlation']) > 0.3]
    strong_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    
    if strong_corrs:
        report.append(f"- {len(strong_corrs)} correlations with medium to large effect size (|r| > 0.3)")
        report.append("  Strongest correlations:")
        for key, corr in strong_corrs[:5]:  # Top 5
            riasec_dim, goal_dim = key.split('_vs_')
            report.append(f"    {riasec_dim} ↔ {goal_dim}: r = {corr:.3f}")
    else:
        report.append("- No correlations with medium to large effect size found")
    
    report.append("")
    
    # Statistical assumptions
    normal_count = sum(1 for result in normality_results.values() if result['p_value'] > 0.05)
    total_vars = len(normality_results)
    report.append(f"STATISTICAL ASSUMPTIONS:")
    report.append(f"- Normality: {normal_count}/{total_vars} variables normally distributed")
    report.append("- Non-parametric alternatives (Spearman) also computed")
    report.append("")
    
    # Implications
    report.append("IMPLICATIONS:")
    report.append("- This analysis explores the relationship between career interests and achievement motivation")
    report.append("- Results can inform career counseling and educational interventions")
    report.append("- Individual profiles show diverse patterns of interests and motivation")
    report.append("")
    
    # Save report
    with open('RIASEC_Goal_Analysis_Summary.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("Summary report saved to: RIASEC_Goal_Analysis_Summary.txt")
    
    return report

def main():
    """Main analysis function"""
    print("RIASEC-GOAL ORIENTATION CROSS-ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Statistical analysis
    correlation_df, p_value_df, correlation_results = calculate_correlations(df)
    normality_results, spearman_results = calculate_comprehensive_statistics(df)
    
    # Pattern analysis
    df_with_patterns = identify_patterns(df, correlation_results)
    
    # Generate summary report
    summary_report = generate_summary_report(correlation_results, normality_results, spearman_results)
    
    print("\n" + "=" * 50)
    print("STATISTICAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nFiles generated:")
    print("- riasec_goal_correlations.csv")
    print("- correlation_p_values.csv")
    print("- combined_riasec_goal_data_with_patterns.csv")
    print("- RIASEC_Goal_Analysis_Summary.txt")
    print("\nNext step: Run visualization script to create plots and charts.")

if __name__ == "__main__":
    main()