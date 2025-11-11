"""
RIASEC-Self Concept Cross-Analysis: Statistical Analysis
Comprehensive correlation and statistical analysis between RIASEC career interests and self-concept dimensions
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro, normaltest
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_combined_data():
    """Load the preprocessed combined dataset"""
    try:
        combined_data = pd.read_csv(r'C:\Users\p2var\vandana\riasec_self_concept_analysis\combined_riasec_self_concept_data.csv')
        print(f"Loaded combined dataset: {len(combined_data)} participants")
        return combined_data
    except FileNotFoundError:
        print("Combined dataset not found. Running preprocessing first...")
        # Import and run preprocessing
        import subprocess
        subprocess.run(['python', 'data_preprocessing.py'], cwd=r'C:\Users\p2var\vandana\riasec_self_concept_analysis')
        combined_data = pd.read_csv(r'C:\Users\p2var\vandana\riasec_self_concept_analysis\combined_riasec_self_concept_data.csv')
        return combined_data

def test_normality(data, variable_name):
    """Test normality of data using Shapiro-Wilk and D'Agostino tests"""
    # Remove any NaN values
    clean_data = data.dropna()
    
    if len(clean_data) < 3:
        return {'shapiro_p': np.nan, 'dagostino_p': np.nan, 'is_normal': False}
    
    # Shapiro-Wilk test (good for small samples)
    shapiro_stat, shapiro_p = shapiro(clean_data)
    
    # D'Agostino test (alternative)
    if len(clean_data) >= 8:
        dagostino_stat, dagostino_p = normaltest(clean_data)
    else:
        dagostino_p = np.nan
    
    # Consider normal if p > 0.05 in either test
    is_normal = shapiro_p > 0.05 or (not np.isnan(dagostino_p) and dagostino_p > 0.05)
    
    return {
        'shapiro_p': shapiro_p,
        'dagostino_p': dagostino_p,
        'is_normal': is_normal
    }

def calculate_correlations(combined_data):
    """Calculate comprehensive correlation analysis between RIASEC and Self-Concept dimensions"""
    print("=== CORRELATION ANALYSIS ===")
    
    # Define dimension columns
    riasec_dimensions = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    self_concept_dimensions = [
        'Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance',
        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt',
        'Sociability', 'Emotional', 'Total_Self_Concept'
    ]
    
    # Filter available columns
    available_riasec = [col for col in riasec_dimensions if col in combined_data.columns]
    available_self_concept = [col for col in self_concept_dimensions if col in combined_data.columns]
    
    print(f"RIASEC dimensions: {available_riasec}")
    print(f"Self-Concept dimensions: {available_self_concept}")
    
    correlation_results = []
    
    # Test normality for all variables
    print("\n=== NORMALITY TESTING ===")
    normality_results = {}
    
    for dim in available_riasec + available_self_concept:
        normality = test_normality(combined_data[dim], dim)
        normality_results[dim] = normality
        print(f"{dim}: Shapiro p={normality['shapiro_p']:.4f}, Normal: {normality['is_normal']}")
    
    # Calculate correlations between all RIASEC and Self-Concept dimensions
    print("\n=== RIASEC-SELF CONCEPT CORRELATIONS ===")
    
    for riasec_dim in available_riasec:
        for self_concept_dim in available_self_concept:
            # Get clean data (remove NaN)
            riasec_data = combined_data[riasec_dim].dropna()
            self_concept_data = combined_data[self_concept_dim].dropna()
            
            # Find common indices
            common_idx = riasec_data.index.intersection(self_concept_data.index)
            if len(common_idx) < 3:
                continue
                
            riasec_clean = riasec_data.loc[common_idx]
            self_concept_clean = self_concept_data.loc[common_idx]
            
            # Choose correlation method based on normality
            riasec_normal = normality_results[riasec_dim]['is_normal']
            self_concept_normal = normality_results[self_concept_dim]['is_normal']
            
            # Calculate both Pearson and Spearman
            pearson_r, pearson_p = pearsonr(riasec_clean, self_concept_clean)
            spearman_r, spearman_p = spearmanr(riasec_clean, self_concept_clean)
            
            # Choose primary correlation method
            if riasec_normal and self_concept_normal:
                primary_r, primary_p, method = pearson_r, pearson_p, "Pearson"
            else:
                primary_r, primary_p, method = spearman_r, spearman_p, "Spearman"
            
            # Effect size interpretation
            if abs(primary_r) < 0.1:
                effect_size = "Negligible"
            elif abs(primary_r) < 0.3:
                effect_size = "Small"
            elif abs(primary_r) < 0.5:
                effect_size = "Medium"
            else:
                effect_size = "Large"
            
            # Significance level
            if primary_p < 0.001:
                significance = "***"
            elif primary_p < 0.01:
                significance = "**"
            elif primary_p < 0.05:
                significance = "*"
            else:
                significance = ""
            
            correlation_results.append({
                'RIASEC_Dimension': riasec_dim,
                'Self_Concept_Dimension': self_concept_dim,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_p,
                'Spearman_r': spearman_r,
                'Spearman_p': spearman_p,
                'Primary_r': primary_r,
                'Primary_p': primary_p,
                'Method': method,
                'Effect_Size': effect_size,
                'Significance': significance,
                'N': len(common_idx)
            })
            
            # Print significant correlations
            if primary_p < 0.05:
                print(f"{riasec_dim} ↔ {self_concept_dim}: r={primary_r:.3f}{significance} (p={primary_p:.4f}, {method}, {effect_size})")
    
    return pd.DataFrame(correlation_results)

def perform_clustering_analysis(combined_data):
    """Perform K-means clustering analysis on combined RIASEC-Self Concept profiles"""
    print("\n=== CLUSTERING ANALYSIS ===")
    
    # Prepare data for clustering
    riasec_columns = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    self_concept_columns = [
        'Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance',
        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt',
        'Sociability', 'Emotional'
    ]
    
    # Filter available columns
    available_riasec = [col for col in riasec_columns if col in combined_data.columns]
    available_self_concept = [col for col in self_concept_columns if col in combined_data.columns]
    
    clustering_data = combined_data[available_riasec + available_self_concept].dropna()
    
    if len(clustering_data) < 3:
        print("Insufficient data for clustering analysis")
        return None
    
    # Standardize the data
    scaler = StandardScaler()
    clustering_data_scaled = scaler.fit_transform(clustering_data)
    
    # Determine optimal number of clusters using elbow method
    max_clusters = min(5, len(clustering_data) - 1)
    inertias = []
    
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(clustering_data_scaled)
        inertias.append(kmeans.inertia_)
    
    # Choose optimal k (elbow point or k=3 if not clear)
    if len(inertias) >= 3:
        # Simple elbow detection
        differences = np.diff(inertias)
        second_differences = np.diff(differences)
        if len(second_differences) > 0:
            optimal_k = np.argmax(second_differences) + 2  # +2 because we start from k=1 and take second diff
        else:
            optimal_k = 3
    else:
        optimal_k = min(3, len(clustering_data))
    
    optimal_k = max(2, min(optimal_k, len(clustering_data) - 1))
    
    print(f"Optimal number of clusters: {optimal_k}")
    
    # Perform final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_data_scaled)
    
    # Add cluster labels to original data
    clustering_results = clustering_data.copy()
    clustering_results['Cluster'] = cluster_labels
    clustering_results['participant'] = combined_data.loc[clustering_data.index, 'participant']
    
    # Analyze cluster characteristics
    print("\n=== CLUSTER PROFILES ===")
    for cluster_id in range(optimal_k):
        cluster_data = clustering_results[clustering_results['Cluster'] == cluster_id]
        print(f"\nCluster {cluster_id + 1} (n={len(cluster_data)}):")
        print(f"Participants: {', '.join(cluster_data['participant'].tolist())}")
        
        # RIASEC profile
        riasec_means = cluster_data[available_riasec].mean()
        print("RIASEC Profile:", riasec_means.round(3).to_dict())
        
        # Self-Concept profile
        self_concept_means = cluster_data[available_self_concept].mean()
        print("Self-Concept Profile:", self_concept_means.round(3).to_dict())
    
    return clustering_results

def perform_pca_analysis(combined_data):
    """Perform Principal Component Analysis on combined data"""
    print("\n=== PRINCIPAL COMPONENT ANALYSIS ===")
    
    # Prepare data
    riasec_columns = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    self_concept_columns = [
        'Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance',
        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt',
        'Sociability', 'Emotional'
    ]
    
    available_riasec = [col for col in riasec_columns if col in combined_data.columns]
    available_self_concept = [col for col in self_concept_columns if col in combined_data.columns]
    
    pca_data = combined_data[available_riasec + available_self_concept].dropna()
    
    if len(pca_data) < 3:
        print("Insufficient data for PCA analysis")
        return None
    
    # Standardize data
    scaler = StandardScaler()
    pca_data_scaled = scaler.fit_transform(pca_data)
    
    # Perform PCA
    n_components = min(len(pca_data.columns), len(pca_data))
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(pca_data_scaled)
    
    # Analyze components
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_).round(3)}")
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=available_riasec + available_self_concept
    )
    
    print("\nComponent Loadings (top 2 components):")
    for i in range(min(2, n_components)):
        print(f"\nPC{i+1} (explains {pca.explained_variance_ratio_[i]:.1%} of variance):")
        sorted_loadings = loadings[f'PC{i+1}'].abs().sort_values(ascending=False)
        for var in sorted_loadings.head(5).index:
            loading = loadings.loc[var, f'PC{i+1}']
            print(f"  {var}: {loading:.3f}")
    
    return pca, loadings, pca_transformed

def individual_profile_analysis(combined_data):
    """Analyze individual participant profiles"""
    print("\n=== INDIVIDUAL PROFILE ANALYSIS ===")
    
    riasec_columns = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    self_concept_columns = [
        'Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance',
        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt',
        'Sociability', 'Emotional', 'Total_Self_Concept'
    ]
    
    available_riasec = [col for col in riasec_columns if col in combined_data.columns]
    available_self_concept = [col for col in self_concept_columns if col in combined_data.columns]
    
    profiles = []
    
    for _, participant_row in combined_data.iterrows():
        participant_name = participant_row['participant']
        
        # RIASEC profile
        riasec_scores = participant_row[available_riasec].astype(float)
        riasec_highest = riasec_scores.idxmax()
        riasec_code = ''.join([dim[0] for dim in riasec_scores.nlargest(3).index])
        
        # Self-concept profile
        self_concept_scores = participant_row[available_self_concept[:-1]].astype(float)  # Exclude total
        self_concept_highest = self_concept_scores.idxmax()
        self_concept_lowest = self_concept_scores.idxmin()
        total_self_concept = participant_row.get('Total_Self_Concept', np.nan)
        
        profiles.append({
            'participant': participant_name,
            'riasec_code': riasec_code,
            'riasec_highest': riasec_highest,
            'riasec_highest_score': riasec_scores[riasec_highest],
            'self_concept_highest': self_concept_highest,
            'self_concept_highest_score': self_concept_scores[self_concept_highest],
            'self_concept_lowest': self_concept_lowest,
            'self_concept_lowest_score': self_concept_scores[self_concept_lowest],
            'total_self_concept': total_self_concept
        })
    
    profiles_df = pd.DataFrame(profiles)
    
    print("Top 5 participant profiles:")
    for i, profile in enumerate(profiles[:5]):
        print(f"\n{i+1}. {profile['participant']}:")
        print(f"   RIASEC Code: {profile['riasec_code']} (Highest: {profile['riasec_highest']} = {profile['riasec_highest_score']:.3f})")
        print(f"   Self-Concept: Highest = {profile['self_concept_highest']} ({profile['self_concept_highest_score']:.3f})")
        print(f"                 Lowest = {profile['self_concept_lowest']} ({profile['self_concept_lowest_score']:.3f})")
        print(f"                 Total = {profile['total_self_concept']:.3f}")
    
    return profiles_df

def generate_summary_statistics(combined_data, correlation_results):
    """Generate comprehensive summary statistics"""
    print("\n=== SUMMARY STATISTICS ===")
    
    # Overall correlations summary
    significant_correlations = correlation_results[correlation_results['Primary_p'] < 0.05]
    medium_large_correlations = significant_correlations[
        significant_correlations['Effect_Size'].isin(['Medium', 'Large'])
    ]
    
    print(f"Total correlations tested: {len(correlation_results)}")
    print(f"Significant correlations (p < 0.05): {len(significant_correlations)}")
    print(f"Medium/Large effect sizes: {len(medium_large_correlations)}")
    
    if len(significant_correlations) > 0:
        print(f"Strongest correlation: r = {significant_correlations.loc[significant_correlations['Primary_r'].abs().idxmax(), 'Primary_r']:.3f}")
        print("Top 5 significant correlations:")
        top_correlations = significant_correlations.nlargest(5, 'Primary_r')
        for _, corr in top_correlations.iterrows():
            print(f"  {corr['RIASEC_Dimension']} ↔ {corr['Self_Concept_Dimension']}: r={corr['Primary_r']:.3f}{corr['Significance']} ({corr['Effect_Size']})")
    
    return {
        'total_correlations': len(correlation_results),
        'significant_correlations': len(significant_correlations),
        'medium_large_effects': len(medium_large_correlations),
        'strongest_correlation': significant_correlations.loc[significant_correlations['Primary_r'].abs().idxmax(), 'Primary_r'] if len(significant_correlations) > 0 else 0
    }

def main():
    """Main analysis function"""
    print("RIASEC-SELF CONCEPT CROSS-ANALYSIS")
    print("=" * 50)
    
    # Load data
    combined_data = load_combined_data()
    
    # Perform analyses
    correlation_results = calculate_correlations(combined_data)
    clustering_results = perform_clustering_analysis(combined_data)
    pca_results = perform_pca_analysis(combined_data)
    profiles = individual_profile_analysis(combined_data)
    summary_stats = generate_summary_statistics(combined_data, correlation_results)
    
    # Save results
    output_dir = r'C:\Users\p2var\vandana\riasec_self_concept_analysis'
    
    correlation_results.to_csv(f'{output_dir}\\correlation_results.csv', index=False)
    print(f"\nCorrelation results saved to: {output_dir}\\correlation_results.csv")
    
    if clustering_results is not None:
        clustering_results.to_csv(f'{output_dir}\\clustering_results.csv', index=False)
        print(f"Clustering results saved to: {output_dir}\\clustering_results.csv")
    
    profiles.to_csv(f'{output_dir}\\individual_profiles.csv', index=False)
    print(f"Individual profiles saved to: {output_dir}\\individual_profiles.csv")
    
    print("\nStatistical analysis completed successfully!")
    print("Next steps: Run visualization script for charts and graphs.")

if __name__ == "__main__":
    main()