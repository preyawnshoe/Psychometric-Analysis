"""
Goal Orientation - Self Concept Cross-Analysis Statistical Analysis
Comprehensive statistical analysis with correlations, clustering, PCA, and individual profiling
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the combined dataset"""
    data_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\combined_goal_self_concept_data.csv'
    return pd.read_csv(data_path)

def test_normality(data, variables):
    """Test normality of variables using Shapiro-Wilk test"""
    print("=== NORMALITY TESTING ===")
    normality_results = {}
    
    for var in variables:
        if var in data.columns:
            var_data = data[var].dropna()
            if len(var_data) > 3:
                statistic, p_value = shapiro(var_data)
                is_normal = p_value > 0.05
                normality_results[var] = {'statistic': statistic, 'p_value': p_value, 'is_normal': is_normal}
                print(f"{var}: Shapiro p={p_value:.4f}, Normal: {is_normal}")
            else:
                normality_results[var] = {'statistic': np.nan, 'p_value': np.nan, 'is_normal': False}
    
    return normality_results

def correlation_analysis(combined_data):
    """Perform comprehensive correlation analysis between Goal Orientation and Self-Concept dimensions"""
    print("\n=== CORRELATION ANALYSIS ===")
    
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance', 
                        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt', 
                        'Sociability', 'Emotional', 'Total_Self_Concept']
    
    all_dims = goal_dims + self_concept_dims
    
    print(f"Goal Orientation dimensions: {goal_dims}")
    print(f"Self-Concept dimensions: {self_concept_dims}")
    
    # Test normality for all dimensions
    normality_results = test_normality(combined_data, all_dims)
    
    # Correlation analysis
    print("\n=== GOAL ORIENTATION-SELF CONCEPT CORRELATIONS ===")
    correlation_results = []
    significant_correlations = []
    
    for goal_dim in goal_dims:
        for sc_dim in self_concept_dims:
            if goal_dim in combined_data.columns and sc_dim in combined_data.columns:
                # Get clean data
                goal_data = combined_data[goal_dim].dropna()
                sc_data = combined_data[sc_dim].dropna()
                
                # Get common indices
                common_idx = combined_data[[goal_dim, sc_dim]].dropna().index
                goal_values = combined_data.loc[common_idx, goal_dim]
                sc_values = combined_data.loc[common_idx, sc_dim]
                
                if len(goal_values) > 3:
                    # Calculate both Pearson and Spearman correlations
                    pearson_r, pearson_p = pearsonr(goal_values, sc_values)
                    spearman_r, spearman_p = spearmanr(goal_values, sc_values)
                    
                    # Choose primary method based on normality
                    goal_normal = normality_results.get(goal_dim, {}).get('is_normal', False)
                    sc_normal = normality_results.get(sc_dim, {}).get('is_normal', False)
                    
                    if goal_normal and sc_normal:
                        primary_r, primary_p = pearson_r, pearson_p
                        method = "Pearson"
                    else:
                        primary_r, primary_p = spearman_r, spearman_p
                        method = "Spearman"
                    
                    # Effect size interpretation
                    effect_size = "Negligible"
                    if abs(primary_r) >= 0.7:
                        effect_size = "Large"
                    elif abs(primary_r) >= 0.5:
                        effect_size = "Medium"
                    elif abs(primary_r) >= 0.3:
                        effect_size = "Small"
                    
                    # Significance
                    significance = ""
                    if primary_p < 0.001:
                        significance = "***"
                    elif primary_p < 0.01:
                        significance = "**"
                    elif primary_p < 0.05:
                        significance = "*"
                    
                    # Store results
                    result = {
                        'Goal_Dimension': goal_dim,
                        'Self_Concept_Dimension': sc_dim,
                        'Pearson_r': pearson_r,
                        'Pearson_p': pearson_p,
                        'Spearman_r': spearman_r,
                        'Spearman_p': spearman_p,
                        'Primary_r': primary_r,
                        'Primary_p': primary_p,
                        'Method': method,
                        'Effect_Size': effect_size,
                        'Significance': significance,
                        'N': len(goal_values)
                    }
                    correlation_results.append(result)
                    
                    # Print significant correlations
                    if primary_p < 0.05:
                        significant_correlations.append(result)
                        print(f"{goal_dim} ↔ {sc_dim}: r={primary_r:.3f}{significance} (p={primary_p:.4f}, {method}, {effect_size})")
    
    # Save correlation results
    corr_df = pd.DataFrame(correlation_results)
    corr_output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\correlation_results.csv'
    corr_df.to_csv(corr_output_path, index=False)
    
    return correlation_results, significant_correlations

def clustering_analysis(combined_data):
    """Perform K-means clustering analysis"""
    print("\n=== CLUSTERING ANALYSIS ===")
    
    # Prepare data for clustering
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance', 
                        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt', 
                        'Sociability', 'Emotional']
    
    all_dims = goal_dims + self_concept_dims
    clustering_data = combined_data[all_dims].select_dtypes(include=[np.number]).dropna()
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(clustering_data)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    K_range = range(1, min(6, len(clustering_data)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_scaled)
        inertias.append(kmeans.inertia_)
    
    # Use elbow method (simple difference approach)
    optimal_k = 2  # Default
    if len(inertias) > 2:
        differences = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        optimal_k = differences.index(max(differences)) + 1
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data_scaled)
    
    # Add cluster labels to data
    clustering_result = combined_data.copy()
    clustering_result['Cluster'] = np.nan
    clustering_result.loc[clustering_data.index, 'Cluster'] = cluster_labels
    
    # Analyze cluster profiles
    print("\n=== CLUSTER PROFILES ===")
    
    for cluster in range(optimal_k):
        cluster_data = clustering_result[clustering_result['Cluster'] == cluster]
        print(f"\nCluster {cluster} (n={len(cluster_data)}):")
        
        participants = cluster_data['participant'].tolist()
        print(f"Participants: {', '.join(participants)}")
        
        # Goal Orientation profile
        goal_means = cluster_data[goal_dims].mean().to_dict()
        print(f"Goal Orientation Profile: {dict((k, round(v, 3)) for k, v in goal_means.items())}")
        
        # Self-Concept profile
        sc_means = cluster_data[self_concept_dims].mean().to_dict()
        print(f"Self-Concept Profile: {dict((k, round(v, 3)) for k, v in sc_means.items())}")
    
    # Save clustering results
    cluster_output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\clustering_results.csv'
    clustering_result.to_csv(cluster_output_path, index=False)
    
    return clustering_result

def pca_analysis(combined_data):
    """Perform Principal Component Analysis"""
    print("\n=== PRINCIPAL COMPONENT ANALYSIS ===")
    
    # Prepare data for PCA
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance', 
                        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt', 
                        'Sociability', 'Emotional']
    
    all_dims = goal_dims + self_concept_dims
    pca_data = combined_data[all_dims].select_dtypes(include=[np.number]).dropna()
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(pca_data)
    
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data_scaled)
    
    # Explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Explained variance ratio: {explained_variance}")
    print(f"Cumulative explained variance: {cumulative_variance}")
    
    # Component loadings
    print("\nComponent Loadings (top 2 components):")
    
    loadings = pca.components_
    feature_names = all_dims
    
    for i in range(min(2, len(loadings))):
        print(f"\nPC{i+1} (explains {explained_variance[i]:.1%} of variance):")
        # Get loadings for this component
        component_loadings = [(feature_names[j], loadings[i][j]) for j in range(len(feature_names))]
        # Sort by absolute loading value
        component_loadings.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Print top 5 loadings
        for feature, loading in component_loadings[:5]:
            print(f"  {feature}: {loading:.3f}")
    
    return pca, pca_result, explained_variance

def individual_profile_analysis(combined_data):
    """Create individual participant profiles"""
    print("\n=== INDIVIDUAL PROFILE ANALYSIS ===")
    
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance', 
                        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt', 
                        'Sociability', 'Emotional', 'Total_Self_Concept']
    
    # Available dimensions in data
    available_goal = [dim for dim in goal_dims if dim in combined_data.columns]
    available_self_concept = [dim for dim in self_concept_dims if dim in combined_data.columns]
    
    profiles = []
    
    for _, participant_row in combined_data.iterrows():
        participant_name = participant_row['participant']
        
        # Goal Orientation profile
        goal_scores = participant_row[available_goal].astype(float)
        goal_highest = goal_scores.idxmax()
        goal_pattern = goal_scores.to_dict()
        
        # Self-concept profile
        self_concept_scores = participant_row[available_self_concept[:-1]].astype(float)  # Exclude total
        self_concept_highest = self_concept_scores.idxmax()
        self_concept_lowest = self_concept_scores.idxmin()
        total_self_concept = participant_row.get('Total_Self_Concept', np.nan)
        
        profile = {
            'participant': participant_name,
            'goal_highest': goal_highest,
            'goal_pattern': str(goal_pattern),
            'self_concept_highest': self_concept_highest,
            'self_concept_lowest': self_concept_lowest,
            'total_self_concept': total_self_concept
        }
        
        # Add individual scores
        for dim in available_goal + available_self_concept:
            profile[dim] = participant_row[dim]
        
        profiles.append(profile)
    
    profiles_df = pd.DataFrame(profiles)
    
    # Display top 5 profiles by total self-concept
    top_profiles = profiles_df.nlargest(5, 'total_self_concept')
    
    print("Top 5 participant profiles:")
    for i, (_, profile) in enumerate(top_profiles.iterrows(), 1):
        print(f"\n{i}. {profile['participant']}:")
        print(f"   Goal Orientation: Highest = {profile['goal_highest']}")
        print(f"   Self-Concept: Highest = {profile['self_concept_highest']}")
        print(f"                 Lowest = {profile['self_concept_lowest']}")
        print(f"                 Total = {profile['total_self_concept']:.3f}")
    
    # Save profiles
    profiles_output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\individual_profiles.csv'
    profiles_df.to_csv(profiles_output_path, index=False)
    
    return profiles_df

def summary_statistics(combined_data, correlation_results):
    """Generate summary statistics"""
    print("\n=== SUMMARY STATISTICS ===")
    
    # Correlation summary
    total_correlations = len(correlation_results)
    significant_correlations = [r for r in correlation_results if r['Primary_p'] < 0.05]
    medium_large_effects = [r for r in significant_correlations if r['Effect_Size'] in ['Medium', 'Large']]
    
    print(f"Total correlations tested: {total_correlations}")
    print(f"Significant correlations (p < 0.05): {len(significant_correlations)}")
    print(f"Medium/Large effect sizes: {len(medium_large_effects)}")
    
    if significant_correlations:
        strongest_corr = max(significant_correlations, key=lambda x: abs(x['Primary_r']))
        print(f"Strongest correlation: r = {strongest_corr['Primary_r']:.3f}")
        
        print("Top 5 significant correlations:")
        sorted_corrs = sorted(significant_correlations, key=lambda x: abs(x['Primary_r']), reverse=True)
        for i, corr in enumerate(sorted_corrs[:5], 1):
            print(f"  {corr['Goal_Dimension']} ↔ {corr['Self_Concept_Dimension']}: r={corr['Primary_r']:.3f}{corr['Significance']} ({corr['Effect_Size']})")

def main():
    """Main statistical analysis function"""
    print("GOAL ORIENTATION-SELF CONCEPT CROSS-ANALYSIS")
    print("=" * 50)
    
    # Load data
    combined_data = load_data()
    print(f"Loaded combined dataset: {len(combined_data)} participants")
    
    # Correlation Analysis
    correlation_results, significant_correlations = correlation_analysis(combined_data)
    
    # Clustering Analysis
    clustering_results = clustering_analysis(combined_data)
    
    # PCA Analysis
    pca, pca_result, explained_variance = pca_analysis(combined_data)
    
    # Individual Profile Analysis
    profiles = individual_profile_analysis(combined_data)
    
    # Summary Statistics
    summary_statistics(combined_data, correlation_results)
    
    # Save file paths
    print(f"\nCorrelation results saved to: C:\\Users\\p2var\\vandana\\goal_self_concept_analysis\\correlation_results.csv")
    print(f"Clustering results saved to: C:\\Users\\p2var\\vandana\\goal_self_concept_analysis\\clustering_results.csv")
    print(f"Individual profiles saved to: C:\\Users\\p2var\\vandana\\goal_self_concept_analysis\\individual_profiles.csv")
    
    print("\nStatistical analysis completed successfully!")
    print("Next steps: Run visualization script for charts and graphs.")

if __name__ == "__main__":
    main()