"""
RIASEC-Self Concept Cross-Analysis Visualizations
Creates comprehensive charts and graphs for the analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the combined dataset"""
    data_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\combined_riasec_self_concept_data.csv'
    return pd.read_csv(data_path)

def create_correlation_heatmap(combined_data):
    """Create correlation heatmap between RIASEC and Self-Concept dimensions"""
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 
                        'Self_Acceptance', 'Worthiness', 'Present_Past_Future', 
                        'Beliefs_Convictions', 'Shame_Guilt', 'Sociability', 'Emotional']
    
    # Calculate correlation matrix
    riasec_data = combined_data[riasec_dims].select_dtypes(include=[np.number])
    self_concept_data = combined_data[self_concept_dims].select_dtypes(include=[np.number])
    
    correlation_matrix = pd.DataFrame(index=riasec_dims, columns=self_concept_dims)
    
    for riasec_dim in riasec_dims:
        for sc_dim in self_concept_dims:
            corr, _ = pearsonr(riasec_data[riasec_dim], self_concept_data[sc_dim])
            correlation_matrix.loc[riasec_dim, sc_dim] = corr
    
    correlation_matrix = correlation_matrix.astype(float)
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    mask = np.abs(correlation_matrix) < 0.3  # Mask weak correlations
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                fmt='.3f',
                mask=mask,
                square=True,
                linewidths=0.5,
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('RIASEC-Self Concept Correlation Matrix\n(Only correlations > |0.3| shown)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Self-Concept Dimensions', fontsize=12, fontweight='bold')
    plt.ylabel('RIASEC Dimensions', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved: {output_path}")
    plt.close()

def create_significant_correlations_plot(combined_data):
    """Create bar plot of significant correlations"""
    
    # Load correlation results
    corr_results = pd.read_csv(r'C:\Users\p2var\vandana\riasec_self_concept_analysis\correlation_results.csv')
    
    # Filter significant correlations
    significant = corr_results[corr_results['Primary_p'] < 0.05].copy()
    significant = significant.sort_values('Primary_r', key=abs, ascending=False)
    
    if len(significant) == 0:
        print("No significant correlations found for plotting")
        return
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    # Color bars based on correlation direction
    colors = ['red' if x < 0 else 'steelblue' for x in significant['Primary_r']]
    
    bars = plt.bar(range(len(significant)), significant['Primary_r'], color=colors, alpha=0.7)
    
    # Add significance stars
    for i, (_, row) in enumerate(significant.iterrows()):
        height = row['Primary_r']
        if row['Primary_p'] < 0.01:
            star = '**'
        else:
            star = '*'
        plt.text(i, height + (0.02 if height > 0 else -0.05), star, 
                ha='center', va='bottom' if height > 0 else 'top', fontsize=12, fontweight='bold')
    
    # Customize plot
    pair_labels = [f"{row['RIASEC_Dimension']} ↔ {row['Self_Concept_Dimension']}" 
                   for _, row in significant.iterrows()]
    
    plt.xticks(range(len(significant)), pair_labels, rotation=45, ha='right')
    plt.ylabel('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    plt.title('Significant RIASEC-Self Concept Correlations\n(* p<0.05, ** p<0.01)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    
    # Add effect size legend
    plt.text(0.02, 0.98, 'Effect Sizes:\n|r| ≥ 0.5 = Large\n|r| ≥ 0.3 = Medium\n|r| ≥ 0.1 = Small',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\significant_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Significant correlations plot saved: {output_path}")
    plt.close()

def create_cluster_profiles_plot(combined_data):
    """Create cluster profiles visualization"""
    
    # Load clustering results
    cluster_results = pd.read_csv(r'C:\Users\p2var\vandana\riasec_self_concept_analysis\clustering_results.csv')
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 
                        'Self_Acceptance', 'Worthiness', 'Present_Past_Future', 
                        'Beliefs_Convictions', 'Shame_Guilt', 'Sociability', 'Emotional']
    
    # Merge with cluster assignments
    combined_with_clusters = combined_data.merge(cluster_results[['participant', 'Cluster']], 
                                                on='participant', how='left')
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # RIASEC profiles
    riasec_cluster_means = combined_with_clusters.groupby('Cluster')[riasec_dims].mean()
    
    x_pos = np.arange(len(riasec_dims))
    width = 0.35
    
    for cluster in riasec_cluster_means.index:
        offset = (cluster - 0.5) * width
        bars = ax1.bar(x_pos + offset, riasec_cluster_means.loc[cluster], 
                      width, label=f'Cluster {cluster+1}', alpha=0.8)
    
    ax1.set_xlabel('RIASEC Dimensions', fontweight='bold')
    ax1.set_ylabel('Mean Score', fontweight='bold')
    ax1.set_title('RIASEC Profiles by Cluster', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([dim[:3] for dim in riasec_dims])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Self-Concept profiles
    sc_cluster_means = combined_with_clusters.groupby('Cluster')[self_concept_dims].mean()
    
    x_pos_sc = np.arange(len(self_concept_dims))
    
    for cluster in sc_cluster_means.index:
        offset = (cluster - 0.5) * width
        bars = ax2.bar(x_pos_sc + offset, sc_cluster_means.loc[cluster], 
                      width, label=f'Cluster {cluster+1}', alpha=0.8)
    
    ax2.set_xlabel('Self-Concept Dimensions', fontweight='bold')
    ax2.set_ylabel('Mean Score', fontweight='bold')
    ax2.set_title('Self-Concept Profiles by Cluster', fontweight='bold')
    ax2.set_xticks(x_pos_sc)
    ax2.set_xticklabels([dim.split('_')[0][:4] for dim in self_concept_dims], rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\cluster_profiles.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cluster profiles plot saved: {output_path}")
    plt.close()

def create_pca_plot(combined_data):
    """Create PCA biplot"""
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 
                        'Self_Acceptance', 'Worthiness', 'Present_Past_Future', 
                        'Beliefs_Convictions', 'Shame_Guilt', 'Sociability', 'Emotional']
    
    all_dims = riasec_dims + self_concept_dims
    data_for_pca = combined_data[all_dims].select_dtypes(include=[np.number])
    
    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_pca)
    
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_scaled)
    
    # Create plot
    plt.figure(figsize=(12, 9))
    
    # Scatter plot of participants
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=100)
    
    # Add participant labels
    for i, participant in enumerate(combined_data['participant']):
        plt.annotate(participant.split()[0][:8], 
                    (pca_result[i, 0], pca_result[i, 1]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add feature vectors
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    for i, feature in enumerate(all_dims):
        plt.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3, 
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        
        # Shorten labels
        if feature in riasec_dims:
            short_label = feature[:3]
        else:
            short_label = feature.split('_')[0][:4]
            
        plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, short_label,
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
    plt.title('PCA Biplot: RIASEC and Self-Concept Dimensions', fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\pca_biplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA biplot saved: {output_path}")
    plt.close()

def create_individual_profiles_radar(combined_data):
    """Create radar charts for top participants"""
    
    # Load individual profiles
    profiles = pd.read_csv(r'C:\Users\p2var\vandana\riasec_self_concept_analysis\individual_profiles.csv')
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    
    # Select top 4 participants by total self-concept score
    top_participants = profiles.nlargest(4, 'total_self_concept')
    
    # Create subplots for radar charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(riasec_dims), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, (_, participant) in enumerate(top_participants.iterrows()):
        ax = axes[idx]
        
        # Get participant data
        participant_data = combined_data[combined_data['participant'] == participant['participant']]
        if len(participant_data) == 0:
            continue
            
        # RIASEC scores
        riasec_scores = participant_data[riasec_dims].iloc[0].tolist()
        riasec_scores += riasec_scores[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, riasec_scores, 'o-', linewidth=2, label='RIASEC Profile')
        ax.fill(angles, riasec_scores, alpha=0.25)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(riasec_dims)
        ax.set_ylim(0, 3)
        ax.set_title(f"{participant['participant'].title()}\n"
                    f"Code: {participant['riasec_code']}, SC: {participant['total_self_concept']:.2f}",
                    pad=20, fontweight='bold')
        ax.grid(True)
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\individual_radar_charts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Individual radar charts saved: {output_path}")
    plt.close()

def create_distribution_plots(combined_data):
    """Create distribution plots for key dimensions"""
    
    # Key dimensions based on significant correlations
    key_dims = ['Investigative', 'Abilities', 'Worthiness', 'Total_Self_Concept']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(key_dims):
        ax = axes[idx]
        
        # Histogram with KDE
        data = combined_data[dim].dropna()
        ax.hist(data, bins=8, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Add mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        ax.set_title(f'Distribution of {dim}', fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\distribution_plots.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plots saved: {output_path}")
    plt.close()

def create_scatter_matrix(combined_data):
    """Create scatter matrix of top correlations"""
    
    # Key variables from significant correlations
    key_vars = ['Investigative', 'Artistic', 'Enterprising', 'Conventional',
               'Abilities', 'Worthiness', 'Self_Acceptance', 'Health_Sex_Appropriateness']
    
    # Create scatter matrix
    fig, axes = plt.subplots(len(key_vars), len(key_vars), figsize=(16, 16))
    
    for i, var1 in enumerate(key_vars):
        for j, var2 in enumerate(key_vars):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(combined_data[var1].dropna(), bins=6, alpha=0.7, color='lightblue')
                ax.set_title(var1.replace('_', ' '), fontsize=10)
            else:
                # Off-diagonal: scatter plot
                ax.scatter(combined_data[var2], combined_data[var1], alpha=0.6)
                
                # Add correlation coefficient
                corr = combined_data[var1].corr(combined_data[var2])
                ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Labels only on edges
            if i == len(key_vars) - 1:
                ax.set_xlabel(var2.replace('_', ' '), fontsize=9)
            if j == 0:
                ax.set_ylabel(var1.replace('_', ' '), fontsize=9)
            
            ax.tick_params(labelsize=8)
    
    plt.suptitle('Scatter Matrix: Key RIASEC and Self-Concept Variables', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\scatter_matrix.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Scatter matrix saved: {output_path}")
    plt.close()

def main():
    """Main visualization function"""
    print("CREATING RIASEC-SELF CONCEPT VISUALIZATIONS")
    print("=" * 50)
    
    # Load data
    combined_data = load_data()
    print(f"Loaded data: {len(combined_data)} participants")
    
    # Import scipy for correlations
    global pearsonr
    from scipy.stats import pearsonr
    
    # Create visualizations
    print("\n1. Creating correlation heatmap...")
    create_correlation_heatmap(combined_data)
    
    print("2. Creating significant correlations plot...")
    create_significant_correlations_plot(combined_data)
    
    print("3. Creating cluster profiles plot...")
    create_cluster_profiles_plot(combined_data)
    
    print("4. Creating PCA biplot...")
    create_pca_plot(combined_data)
    
    print("5. Creating individual radar charts...")
    create_individual_profiles_radar(combined_data)
    
    print("6. Creating distribution plots...")
    create_distribution_plots(combined_data)
    
    print("7. Creating scatter matrix...")
    create_scatter_matrix(combined_data)
    
    print("\n" + "=" * 50)
    print("All visualizations created successfully!")
    print("Next step: Create comprehensive PDF report")

if __name__ == "__main__":
    main()