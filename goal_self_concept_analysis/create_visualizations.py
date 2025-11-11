"""
Goal Orientation - Self Concept Cross-Analysis Visualizations
Creates comprehensive charts and graphs for the analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load the combined dataset"""
    data_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\combined_goal_self_concept_data.csv'
    return pd.read_csv(data_path)

def create_correlation_heatmap(combined_data):
    """Create correlation heatmap between Goal Orientation and Self-Concept dimensions"""
    
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 
                        'Self_Acceptance', 'Worthiness', 'Present_Past_Future', 
                        'Beliefs_Convictions', 'Shame_Guilt', 'Sociability', 'Emotional']
    
    # Calculate correlation matrix
    goal_data = combined_data[goal_dims].select_dtypes(include=[np.number])
    self_concept_data = combined_data[self_concept_dims].select_dtypes(include=[np.number])
    
    correlation_matrix = pd.DataFrame(index=goal_dims, columns=self_concept_dims)
    
    for goal_dim in goal_dims:
        for sc_dim in self_concept_dims:
            corr, _ = pearsonr(goal_data[goal_dim], self_concept_data[sc_dim])
            correlation_matrix.loc[goal_dim, sc_dim] = corr
    
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
    
    plt.title('Goal Orientation-Self Concept Correlation Matrix\n(Only correlations > |0.3| shown)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Self-Concept Dimensions', fontsize=12, fontweight='bold')
    plt.ylabel('Goal Orientation Dimensions', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Correlation heatmap saved: {output_path}")
    plt.close()

def create_significant_correlations_plot(combined_data):
    """Create bar plot of significant correlations"""
    
    # Load correlation results
    corr_results = pd.read_csv(r'C:\Users\p2var\vandana\goal_self_concept_analysis\correlation_results.csv')
    
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
    pair_labels = [f"{row['Goal_Dimension'].replace('_', ' ')} ↔ {row['Self_Concept_Dimension'].replace('_', ' ')}" 
                   for _, row in significant.iterrows()]
    
    plt.xticks(range(len(significant)), pair_labels, rotation=45, ha='right')
    plt.ylabel('Correlation Coefficient (r)', fontsize=12, fontweight='bold')
    plt.title('Significant Goal Orientation-Self Concept Correlations\n(* p<0.05, ** p<0.01)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(axis='y', alpha=0.3)
    
    # Add effect size legend
    plt.text(0.02, 0.98, 'Effect Sizes:\n|r| ≥ 0.7 = Large\n|r| ≥ 0.5 = Medium\n|r| ≥ 0.3 = Small',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\significant_correlations.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Significant correlations plot saved: {output_path}")
    plt.close()

def create_goal_orientation_profiles(combined_data):
    """Create goal orientation profiles visualization"""
    
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Calculate mean scores for each dimension
    mean_scores = combined_data[goal_dims].mean()
    std_scores = combined_data[goal_dims].std()
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    
    x_pos = np.arange(len(goal_dims))
    bars = plt.bar(x_pos, mean_scores, yerr=std_scores, capsize=5, alpha=0.7, color='skyblue')
    
    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(mean_scores, std_scores)):
        plt.text(i, mean_val + std_val + 0.05, f'{mean_val:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Goal Orientation Dimensions', fontsize=12, fontweight='bold')
    plt.ylabel('Mean Score (1-5 scale)', fontsize=12, fontweight='bold')
    plt.title('Goal Orientation Profile Across All Participants', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x_pos, [dim.replace('_', ' ') for dim in goal_dims], rotation=45, ha='right')
    plt.ylim(0, 5.5)
    plt.grid(axis='y', alpha=0.3)
    
    # Add interpretation text
    plt.text(0.02, 0.98, 'Higher scores indicate stronger orientation toward that goal type',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\goal_orientation_profiles.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Goal orientation profiles plot saved: {output_path}")
    plt.close()

def create_self_concept_profiles(combined_data):
    """Create self-concept profiles visualization"""
    
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 
                        'Self_Acceptance', 'Worthiness', 'Present_Past_Future', 
                        'Beliefs_Convictions', 'Shame_Guilt', 'Sociability', 'Emotional']
    
    # Calculate mean scores for each dimension
    mean_scores = combined_data[self_concept_dims].mean()
    std_scores = combined_data[self_concept_dims].std()
    
    # Create horizontal bar plot for better readability
    plt.figure(figsize=(12, 10))
    
    y_pos = np.arange(len(self_concept_dims))
    bars = plt.barh(y_pos, mean_scores, xerr=std_scores, capsize=5, alpha=0.7, color='lightcoral')
    
    # Add value labels on bars
    for i, (mean_val, std_val) in enumerate(zip(mean_scores, std_scores)):
        plt.text(mean_val + std_val + 0.05, i, f'{mean_val:.2f}', 
                ha='left', va='center', fontweight='bold')
    
    plt.ylabel('Self-Concept Dimensions', fontsize=12, fontweight='bold')
    plt.xlabel('Mean Score (1-5 scale)', fontsize=12, fontweight='bold')
    plt.title('Self-Concept Profile Across All Participants', fontsize=14, fontweight='bold', pad=20)
    plt.yticks(y_pos, [dim.replace('_', ' ') for dim in self_concept_dims])
    plt.xlim(0, 5.5)
    plt.grid(axis='x', alpha=0.3)
    
    # Add interpretation text
    plt.text(0.02, 0.98, 'Higher scores indicate more positive self-concept in that dimension',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\self_concept_profiles.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Self-concept profiles plot saved: {output_path}")
    plt.close()

def create_pca_biplot(combined_data):
    """Create PCA biplot"""
    
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    self_concept_dims = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 
                        'Self_Acceptance', 'Worthiness', 'Present_Past_Future', 
                        'Beliefs_Convictions', 'Shame_Guilt', 'Sociability', 'Emotional']
    
    all_dims = goal_dims + self_concept_dims
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
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=100, color='blue')
    
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
        
        # Shorten labels for readability
        short_label = feature.replace('_', ' ')[:8]
        if len(short_label) > 8:
            short_label = short_label[:6] + ".."
            
        plt.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, short_label,
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontweight='bold')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontweight='bold')
    plt.title('PCA Biplot: Goal Orientation and Self-Concept Dimensions', fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\pca_biplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"PCA biplot saved: {output_path}")
    plt.close()

def create_individual_radar_charts(combined_data):
    """Create radar charts for top participants"""
    
    # Load individual profiles
    profiles = pd.read_csv(r'C:\Users\p2var\vandana\goal_self_concept_analysis\individual_profiles.csv')
    
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Select top 4 participants by total self-concept score
    top_participants = profiles.nlargest(4, 'total_self_concept')
    
    # Create subplots for radar charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()
    
    # Angles for radar chart
    angles = np.linspace(0, 2 * np.pi, len(goal_dims), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for idx, (_, participant) in enumerate(top_participants.iterrows()):
        ax = axes[idx]
        
        # Get participant data
        participant_data = combined_data[combined_data['participant'] == participant['participant']]
        if len(participant_data) == 0:
            continue
            
        # Goal orientation scores
        goal_scores = participant_data[goal_dims].iloc[0].tolist()
        goal_scores += goal_scores[:1]  # Complete the circle
        
        # Plot
        ax.plot(angles, goal_scores, 'o-', linewidth=2, label='Goal Orientation Profile')
        ax.fill(angles, goal_scores, alpha=0.25)
        
        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([dim.replace('_', ' ') for dim in goal_dims])
        ax.set_ylim(0, 5)
        ax.set_title(f"{participant['participant'].title()}\n"
                    f"Total SC: {participant['total_self_concept']:.2f}",
                    pad=20, fontweight='bold')
        ax.grid(True)
    
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\individual_radar_charts.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Individual radar charts saved: {output_path}")
    plt.close()

def create_scatter_plots_key_correlations(combined_data):
    """Create scatter plots for key significant correlations"""
    
    # Key correlations from the analysis
    key_correlations = [
        ('Mastery_Approach', 'Worthiness'),
        ('Mastery_Approach', 'Abilities'),
        ('Mastery_Avoidance', 'Self_Confidence'),
        ('Mastery_Approach', 'Beliefs_Convictions')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (goal_dim, sc_dim) in enumerate(key_correlations):
        ax = axes[idx]
        
        x = combined_data[goal_dim]
        y = combined_data[sc_dim]
        
        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=100)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8)
        
        # Calculate correlation
        corr, p_val = pearsonr(x, y)
        
        ax.set_xlabel(goal_dim.replace('_', ' '), fontweight='bold')
        ax.set_ylabel(sc_dim.replace('_', ' '), fontweight='bold')
        ax.set_title(f'{goal_dim.replace("_", " ")} vs {sc_dim.replace("_", " ")}\nr = {corr:.3f}', 
                    fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Key Significant Correlations: Goal Orientation ↔ Self-Concept', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\key_correlations_scatter.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Key correlations scatter plots saved: {output_path}")
    plt.close()

def create_distribution_plots(combined_data):
    """Create distribution plots for all dimensions"""
    
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, dim in enumerate(goal_dims):
        ax = axes[idx]
        
        data = combined_data[dim].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=8, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        
        # Add mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        ax.set_title(f'Distribution of {dim.replace("_", " ")}', fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Goal Orientation Dimensions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\goal_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Goal orientation distributions saved: {output_path}")
    plt.close()

def main():
    """Main visualization function"""
    print("CREATING GOAL ORIENTATION-SELF CONCEPT VISUALIZATIONS")
    print("=" * 60)
    
    # Load data
    combined_data = load_data()
    print(f"Loaded data: {len(combined_data)} participants")
    
    # Create visualizations
    print("\n1. Creating correlation heatmap...")
    create_correlation_heatmap(combined_data)
    
    print("2. Creating significant correlations plot...")
    create_significant_correlations_plot(combined_data)
    
    print("3. Creating goal orientation profiles...")
    create_goal_orientation_profiles(combined_data)
    
    print("4. Creating self-concept profiles...")
    create_self_concept_profiles(combined_data)
    
    print("5. Creating PCA biplot...")
    create_pca_biplot(combined_data)
    
    print("6. Creating individual radar charts...")
    create_individual_radar_charts(combined_data)
    
    print("7. Creating key correlations scatter plots...")
    create_scatter_plots_key_correlations(combined_data)
    
    print("8. Creating distribution plots...")
    create_distribution_plots(combined_data)
    
    print("\n" + "=" * 60)
    print("All visualizations created successfully!")
    print("Next step: Create comprehensive PDF report")

if __name__ == "__main__":
    main()