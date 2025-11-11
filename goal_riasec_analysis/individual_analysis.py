import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the combined dataset with patterns"""
    df = pd.read_csv('combined_riasec_goal_data_with_patterns.csv')
    return df

def calculate_individual_statistics(df):
    """Calculate detailed statistics for each participant"""
    print("Calculating individual participant statistics...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    individual_stats = []
    
    for idx, row in df.iterrows():
        participant = row['participant']
        
        # RIASEC statistics
        riasec_scores = row[riasec_dims]
        riasec_mean = riasec_scores.mean()
        riasec_std = riasec_scores.std()
        riasec_max = riasec_scores.max()
        riasec_min = riasec_scores.min()
        riasec_range = riasec_max - riasec_min
        
        # Goal orientation statistics
        goal_scores = row[goal_dims]
        goal_mean = goal_scores.mean()
        goal_std = goal_scores.std()
        goal_max = goal_scores.max()
        goal_min = goal_scores.min()
        goal_range = goal_max - goal_min
        
        # Calculate profile consistency (inverse of standard deviation)
        riasec_consistency = 1 / (riasec_std + 0.001)  # Add small value to avoid division by zero
        goal_consistency = 1 / (goal_std + 0.001)
        
        # Approach vs Avoidance tendencies
        approach_score = (row['Performance_Approach'] + row['Mastery_Approach']) / 2
        avoidance_score = (row['Performance_Avoidance'] + row['Mastery_Avoidance']) / 2
        approach_vs_avoidance = approach_score - avoidance_score
        
        # Performance vs Mastery tendencies
        performance_score = (row['Performance_Approach'] + row['Performance_Avoidance']) / 2
        mastery_score = (row['Mastery_Approach'] + row['Mastery_Avoidance']) / 2
        performance_vs_mastery = performance_score - mastery_score
        
        individual_stats.append({
            'participant': participant,
            'riasec_mean': riasec_mean,
            'riasec_std': riasec_std,
            'riasec_range': riasec_range,
            'riasec_consistency': riasec_consistency,
            'goal_mean': goal_mean,
            'goal_std': goal_std,
            'goal_range': goal_range,
            'goal_consistency': goal_consistency,
            'approach_score': approach_score,
            'avoidance_score': avoidance_score,
            'approach_vs_avoidance': approach_vs_avoidance,
            'performance_score': performance_score,
            'mastery_score': mastery_score,
            'performance_vs_mastery': performance_vs_mastery,
            'dominant_riasec': row['Dominant_RIASEC'],
            'dominant_goal': row['Dominant_Goal']
        })
    
    stats_df = pd.DataFrame(individual_stats)
    stats_df.to_csv('individual_participant_statistics.csv', index=False)
    
    return stats_df

def perform_clustering_analysis(df):
    """Perform clustering analysis on participant profiles"""
    print("Performing clustering analysis...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Prepare data for clustering
    clustering_data = df[riasec_dims + goal_dims].copy()
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Determine optimal number of clusters using elbow method
    inertias = []
    k_range = range(2, 8)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
    
    # Use 3 clusters as optimal (for this sample size)
    optimal_k = 3
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to dataframe
    df['Cluster'] = cluster_labels
    
    # Analyze clusters
    cluster_analysis = []
    for cluster_id in range(optimal_k):
        cluster_participants = df[df['Cluster'] == cluster_id]
        
        cluster_info = {
            'cluster_id': cluster_id,
            'size': len(cluster_participants),
            'participants': cluster_participants['participant'].tolist(),
            'riasec_means': cluster_participants[riasec_dims].mean().to_dict(),
            'goal_means': cluster_participants[goal_dims].mean().to_dict(),
            'dominant_riasec': cluster_participants['Dominant_RIASEC'].mode().iloc[0] if len(cluster_participants) > 0 else 'N/A',
            'dominant_goal': cluster_participants['Dominant_Goal'].mode().iloc[0] if len(cluster_participants) > 0 else 'N/A'
        }
        cluster_analysis.append(cluster_info)
    
    return df, cluster_analysis, scaled_data, scaler

def create_individual_profile_plots(df, stats_df):
    """Create individual profile visualizations"""
    print("Creating individual profile plots...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Create subplot for each participant
    n_participants = len(df)
    n_cols = 4
    n_rows = (n_participants + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        participant = row['participant']
        
        # Create radar chart-like visualization
        categories = riasec_dims + goal_dims
        values = [row[dim] for dim in categories]
        
        # Create bar plot
        colors = ['skyblue'] * len(riasec_dims) + ['orange'] * len(goal_dims)
        bars = ax.bar(range(len(categories)), values, color=colors, alpha=0.7)
        
        # Highlight dominant dimensions
        riasec_max_idx = riasec_dims.index(row['Dominant_RIASEC'])
        goal_max_idx = len(riasec_dims) + goal_dims.index(row['Dominant_Goal'])
        
        bars[riasec_max_idx].set_color('blue')
        bars[goal_max_idx].set_color('red')
        
        ax.set_title(f'{participant}', fontweight='bold', fontsize=10)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([cat[:4] for cat in categories], rotation=45, fontsize=8)
        ax.set_ylim(0, 4)
        ax.grid(True, alpha=0.3)
        
        # Add consistency info
        riasec_consistency = stats_df[stats_df['participant'] == participant]['riasec_consistency'].iloc[0]
        goal_consistency = stats_df[stats_df['participant'] == participant]['goal_consistency'].iloc[0]
        ax.text(0.02, 0.98, f'R: {riasec_consistency:.2f}\\nG: {goal_consistency:.2f}', 
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Hide empty subplots
    for idx in range(n_participants, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Individual Participant Profiles\\n(Blue=Dominant RIASEC, Red=Dominant Goal, Consistency scores shown)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('individual_participant_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Individual profiles saved as 'individual_participant_profiles.png'")

def create_clustering_visualization(df, cluster_analysis, scaled_data):
    """Create clustering visualization"""
    print("Creating clustering visualization...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # PCA visualization
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for cluster_id in range(len(cluster_analysis)):
        cluster_mask = df['Cluster'] == cluster_id
        ax1.scatter(pca_data[cluster_mask, 0], pca_data[cluster_mask, 1], 
                   c=colors[cluster_id], label=f'Cluster {cluster_id}', 
                   s=100, alpha=0.7, edgecolors='black')
    
    # Add participant labels
    for i, participant in enumerate(df['participant']):
        ax1.annotate(participant[:6], (pca_data[i, 0], pca_data[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('Participant Clusters (PCA Visualization)', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cluster characteristics heatmap
    cluster_means = []
    for cluster_info in cluster_analysis:
        means = list(cluster_info['riasec_means'].values()) + list(cluster_info['goal_means'].values())
        cluster_means.append(means)
    
    cluster_df = pd.DataFrame(cluster_means, 
                             columns=riasec_dims + goal_dims,
                             index=[f'Cluster {i}' for i in range(len(cluster_analysis))])
    
    sns.heatmap(cluster_df, annot=True, cmap='viridis', fmt='.2f', ax=ax2)
    ax2.set_title('Cluster Characteristics', fontweight='bold')
    ax2.set_xlabel('Dimensions')
    
    # Cluster composition
    cluster_sizes = [info['size'] for info in cluster_analysis]
    cluster_labels = [f'Cluster {i}\\n({info["size"]} participants)' for i, info in enumerate(cluster_analysis)]
    
    ax3.pie(cluster_sizes, labels=cluster_labels, autopct='%1.1f%%', startangle=90, colors=colors[:len(cluster_analysis)])
    ax3.set_title('Cluster Size Distribution', fontweight='bold')
    
    # Dominant types by cluster
    dominant_riasec_by_cluster = {}
    dominant_goal_by_cluster = {}
    
    for cluster_info in cluster_analysis:
        cluster_id = cluster_info['cluster_id']
        dominant_riasec_by_cluster[f'Cluster {cluster_id}'] = cluster_info['dominant_riasec']
        dominant_goal_by_cluster[f'Cluster {cluster_id}'] = cluster_info['dominant_goal']
    
    # Create text summary
    ax4.axis('off')
    summary_text = "CLUSTER ANALYSIS SUMMARY:\\n\\n"
    
    for i, cluster_info in enumerate(cluster_analysis):
        summary_text += f"CLUSTER {i} ({cluster_info['size']} participants):\\n"
        summary_text += f"  Participants: {', '.join(cluster_info['participants'][:3])}{'...' if len(cluster_info['participants']) > 3 else ''}\\n"
        summary_text += f"  Dominant RIASEC: {cluster_info['dominant_riasec']}\\n"
        summary_text += f"  Dominant Goal: {cluster_info['dominant_goal']}\\n\\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Clustering analysis saved as 'clustering_analysis.png'")

def generate_individual_reports(df, stats_df, cluster_analysis):
    """Generate detailed individual participant reports"""
    print("Generating individual participant reports...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    reports = []
    
    for idx, row in df.iterrows():
        participant = row['participant']
        stats_row = stats_df[stats_df['participant'] == participant].iloc[0]
        
        report = []
        report.append(f"INDIVIDUAL ANALYSIS REPORT: {participant.upper()}")
        report.append("=" * 60)
        report.append("")
        
        # Basic profile
        report.append("PROFILE SUMMARY:")
        report.append(f"  Dominant RIASEC Type: {row['Dominant_RIASEC']}")
        report.append(f"  Dominant Goal Orientation: {row['Dominant_Goal']}")
        report.append(f"  Cluster Assignment: Cluster {row['Cluster']}")
        report.append("")
        
        # RIASEC analysis
        report.append("RIASEC CAREER INTERESTS:")
        riasec_scores = [(dim, row[dim]) for dim in riasec_dims]
        riasec_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (dim, score) in enumerate(riasec_scores):
            rank = "Highest" if i == 0 else "Lowest" if i == len(riasec_scores)-1 else f"#{i+1}"
            report.append(f"  {dim}: {score:.3f} ({rank})")
        
        report.append(f"  Mean RIASEC Score: {stats_row['riasec_mean']:.3f}")
        report.append(f"  Profile Consistency: {stats_row['riasec_consistency']:.3f}")
        report.append("")
        
        # Goal orientation analysis
        report.append("GOAL ORIENTATION:")
        goal_scores = [(dim, row[dim]) for dim in goal_dims]
        goal_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (dim, score) in enumerate(goal_scores):
            rank = "Highest" if i == 0 else "Lowest" if i == len(goal_scores)-1 else f"#{i+1}"
            report.append(f"  {dim}: {score:.3f} ({rank})")
        
        report.append(f"  Mean Goal Score: {stats_row['goal_mean']:.3f}")
        report.append(f"  Profile Consistency: {stats_row['goal_consistency']:.3f}")
        report.append("")
        
        # Motivation patterns
        report.append("MOTIVATION PATTERNS:")
        report.append(f"  Approach vs Avoidance: {stats_row['approach_vs_avoidance']:.3f}")
        if stats_row['approach_vs_avoidance'] > 0:
            report.append("    → More approach-oriented (seeks positive outcomes)")
        else:
            report.append("    → More avoidance-oriented (avoids negative outcomes)")
        
        report.append(f"  Performance vs Mastery: {stats_row['performance_vs_mastery']:.3f}")
        if stats_row['performance_vs_mastery'] > 0:
            report.append("    → More performance-oriented (competitive, comparative)")
        else:
            report.append("    → More mastery-oriented (learning, skill development)")
        report.append("")
        
        # Cluster characteristics
        cluster_id = row['Cluster']
        cluster_info = cluster_analysis[cluster_id]
        
        report.append(f"CLUSTER {cluster_id} CHARACTERISTICS:")
        report.append(f"  Cluster Size: {cluster_info['size']} participants")
        report.append(f"  Typical RIASEC: {cluster_info['dominant_riasec']}")
        report.append(f"  Typical Goal: {cluster_info['dominant_goal']}")
        report.append(f"  Cluster Members: {', '.join(cluster_info['participants'])}")
        report.append("")
        
        # Personalized recommendations
        report.append("PERSONALIZED INSIGHTS:")
        
        # Career recommendations based on RIASEC
        riasec_careers = {
            'Realistic': 'engineering, trades, agriculture, technology',
            'Investigative': 'research, science, analysis, medicine',
            'Artistic': 'creative arts, design, writing, media',
            'Social': 'education, counseling, healthcare, social work',
            'Enterprising': 'business, sales, leadership, entrepreneurship',
            'Conventional': 'administration, accounting, data management, organization'
        }
        
        dominant_riasec = row['Dominant_RIASEC']
        report.append(f"  Career Fields: Consider {riasec_careers[dominant_riasec]}")
        
        # Learning recommendations based on goal orientation
        if row['Mastery_Approach'] > row['Performance_Approach']:
            report.append("  Learning Style: Focus on deep understanding and skill mastery")
        else:
            report.append("  Learning Style: Thrives in competitive environments with clear benchmarks")
        
        if stats_row['approach_vs_avoidance'] > 0.5:
            report.append("  Motivation: Set challenging goals and focus on achievement")
        elif stats_row['approach_vs_avoidance'] < -0.5:
            report.append("  Motivation: Create supportive environments and reduce performance pressure")
        else:
            report.append("  Motivation: Balanced approach with both goals and safety nets")
        
        report.append("")
        report.append("=" * 60)
        
        reports.append('\\n'.join(report))
    
    # Save all reports
    full_report = '\\n\\n'.join(reports)
    with open('Individual_Participant_Reports.txt', 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print("Individual reports saved to 'Individual_Participant_Reports.txt'")
    
    return reports

def create_comparison_matrix(df):
    """Create participant comparison matrix"""
    print("Creating participant comparison matrix...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    # Create comparison heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # RIASEC comparison
    riasec_data = df.set_index('participant')[riasec_dims].T
    sns.heatmap(riasec_data, annot=True, fmt='.2f', cmap='viridis', ax=ax1)
    ax1.set_title('RIASEC Profiles Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Participants')
    ax1.set_ylabel('RIASEC Dimensions')
    
    # Goal orientation comparison
    goal_data = df.set_index('participant')[goal_dims].T
    sns.heatmap(goal_data, annot=True, fmt='.2f', cmap='plasma', ax=ax2)
    ax2.set_title('Goal Orientation Profiles Comparison', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Participants')
    ax2.set_ylabel('Goal Orientation Dimensions')
    
    plt.tight_layout()
    plt.savefig('participant_comparison_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison matrix saved as 'participant_comparison_matrix.png'")

def main():
    """Main individual analysis function"""
    print("INDIVIDUAL PARTICIPANT ANALYSIS")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Calculate individual statistics
    stats_df = calculate_individual_statistics(df)
    
    # Perform clustering
    df_with_clusters, cluster_analysis, scaled_data, scaler = perform_clustering_analysis(df)
    
    # Create visualizations
    create_individual_profile_plots(df_with_clusters, stats_df)
    create_clustering_visualization(df_with_clusters, cluster_analysis, scaled_data)
    create_comparison_matrix(df_with_clusters)
    
    # Generate reports
    reports = generate_individual_reports(df_with_clusters, stats_df, cluster_analysis)
    
    # Save updated dataset with clusters
    df_with_clusters.to_csv('combined_riasec_goal_data_final.csv', index=False)
    
    print("\\n" + "=" * 50)
    print("INDIVIDUAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\\nGenerated files:")
    print("- individual_participant_statistics.csv")
    print("- individual_participant_profiles.png") 
    print("- clustering_analysis.png")
    print("- participant_comparison_matrix.png")
    print("- Individual_Participant_Reports.txt")
    print("- combined_riasec_goal_data_final.csv")
    
    # Print summary
    print(f"\\nAnalysis Summary:")
    print(f"- {len(df)} participants analyzed")
    print(f"- {len(cluster_analysis)} clusters identified")
    print(f"- Individual reports generated for all participants")
    print(f"- Comprehensive statistical and visual analysis completed")

if __name__ == "__main__":
    main()