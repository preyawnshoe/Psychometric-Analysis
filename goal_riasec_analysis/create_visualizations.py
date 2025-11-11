import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def setup_plot_style():
    """Set up consistent plot styling"""
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def load_data():
    """Load all required datasets"""
    df = pd.read_csv('combined_riasec_goal_data_with_patterns.csv')
    correlations = pd.read_csv('riasec_goal_correlations.csv', index_col=0)
    p_values = pd.read_csv('correlation_p_values.csv', index_col=0)
    
    return df, correlations, p_values

def create_correlation_heatmap(correlations, p_values):
    """Create a comprehensive correlation heatmap"""
    print("Creating correlation heatmap...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Correlation heatmap
    mask_corr = np.zeros_like(correlations, dtype=bool)
    im1 = sns.heatmap(correlations, annot=True, cmap='RdBu_r', center=0, 
                      square=True, fmt='.3f', cbar_kws={"shrink": .8},
                      ax=ax1, vmin=-0.8, vmax=0.8)
    ax1.set_title('RIASEC - Goal Orientation Correlations\n(Pearson Correlation Coefficients)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Goal Orientation Dimensions', fontweight='bold')
    ax1.set_ylabel('RIASEC Dimensions', fontweight='bold')
    
    # Create significance overlay
    for i in range(len(correlations.index)):
        for j in range(len(correlations.columns)):
            p_val = p_values.iloc[i, j]
            if p_val < 0.001:
                significance = '***'
            elif p_val < 0.01:
                significance = '**'
            elif p_val < 0.05:
                significance = '*'
            else:
                significance = ''
            
            if significance:
                ax1.text(j + 0.5, i + 0.8, significance, ha='center', va='center',
                        fontsize=12, fontweight='bold', color='black')
    
    # P-values heatmap
    im2 = sns.heatmap(p_values, annot=True, cmap='Reds', square=True, 
                      fmt='.3f', cbar_kws={"shrink": .8}, ax=ax2)
    ax2.set_title('Statistical Significance\n(P-values)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Goal Orientation Dimensions', fontweight='bold')
    ax2.set_ylabel('RIASEC Dimensions', fontweight='bold')
    
    # Add significance threshold lines
    ax2.axhline(y=0, color='yellow', linewidth=2, alpha=0.7)  # p < 0.05 threshold
    
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Correlation heatmap saved as 'correlation_heatmap.png'")

def create_scatter_plots(df):
    """Create scatter plots for significant correlations"""
    print("Creating scatter plots for significant correlations...")
    
    # Define significant correlations
    significant_pairs = [
        ('Conventional', 'Mastery_Approach'),
        ('Investigative', 'Mastery_Approach'),
        ('Realistic', 'Mastery_Approach')
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (riasec_dim, goal_dim) in enumerate(significant_pairs):
        ax = axes[i]
        
        # Create scatter plot
        x = df[riasec_dim]
        y = df[goal_dim]
        
        ax.scatter(x, y, alpha=0.7, s=100, edgecolors='black', linewidth=1)
        
        # Add trend line
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        corr = np.corrcoef(x, y)[0, 1]
        
        # Add correlation info
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                fontsize=12, fontweight='bold')
        
        ax.set_xlabel(riasec_dim, fontweight='bold')
        ax.set_ylabel(goal_dim, fontweight='bold')
        ax.set_title(f'{riasec_dim} vs {goal_dim}', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Significant RIASEC-Goal Orientation Correlations', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('significant_correlations_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Scatter plots saved as 'significant_correlations_scatter.png'")

def create_distribution_plots(df):
    """Create distribution plots for RIASEC and Goal dimensions"""
    print("Creating distribution plots...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # RIASEC distributions
    df[riasec_dims].plot(kind='box', ax=ax1)
    ax1.set_title('RIASEC Dimensions Distribution', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Score', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Goal orientation distributions
    df[goal_dims].plot(kind='box', ax=ax2)
    ax2.set_title('Goal Orientation Dimensions Distribution', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # RIASEC violin plots
    riasec_melted = df[riasec_dims].melt()
    sns.violinplot(data=riasec_melted, x='variable', y='value', ax=ax3)
    ax3.set_title('RIASEC Dimensions - Density Distribution', fontweight='bold', fontsize=14)
    ax3.set_xlabel('RIASEC Dimension', fontweight='bold')
    ax3.set_ylabel('Score', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # Goal orientation violin plots
    goal_melted = df[goal_dims].melt()
    sns.violinplot(data=goal_melted, x='variable', y='value', ax=ax4)
    ax4.set_title('Goal Orientation - Density Distribution', fontweight='bold', fontsize=14)
    ax4.set_xlabel('Goal Orientation Dimension', fontweight='bold')
    ax4.set_ylabel('Score', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Distribution plots saved as 'distribution_plots.png'")

def create_participant_profiles_plot(df):
    """Create individual participant profile visualization"""
    print("Creating participant profiles plot...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    # RIASEC profiles
    df_riasec = df.set_index('participant')[riasec_dims]
    df_riasec.T.plot(kind='line', marker='o', ax=ax1, linewidth=2, markersize=6)
    ax1.set_title('Individual RIASEC Profiles', fontweight='bold', fontsize=14)
    ax1.set_ylabel('RIASEC Score', fontweight='bold')
    ax1.set_xlabel('RIASEC Dimensions', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # Goal orientation profiles
    df_goal = df.set_index('participant')[goal_dims]
    df_goal.T.plot(kind='line', marker='s', ax=ax2, linewidth=2, markersize=6)
    ax2.set_title('Individual Goal Orientation Profiles', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Goal Orientation Score', fontweight='bold')
    ax2.set_xlabel('Goal Orientation Dimensions', fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('participant_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Participant profiles saved as 'participant_profiles.png'")

def create_dominant_types_visualization(df):
    """Create visualization of dominant types"""
    print("Creating dominant types visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # RIASEC dominant types pie chart
    riasec_counts = df['Dominant_RIASEC'].value_counts()
    colors1 = sns.color_palette("Set3", len(riasec_counts))
    wedges1, texts1, autotexts1 = ax1.pie(riasec_counts.values, labels=riasec_counts.index, 
                                          autopct='%1.1f%%', colors=colors1, startangle=90)
    ax1.set_title('Dominant RIASEC Types Distribution', fontweight='bold', fontsize=14)
    
    # Goal orientation dominant types pie chart
    goal_counts = df['Dominant_Goal'].value_counts()
    colors2 = sns.color_palette("Set2", len(goal_counts))
    wedges2, texts2, autotexts2 = ax2.pie(goal_counts.values, labels=goal_counts.index, 
                                          autopct='%1.1f%%', colors=colors2, startangle=90)
    ax2.set_title('Dominant Goal Orientation Distribution', fontweight='bold', fontsize=14)
    
    # Cross-tabulation heatmap
    crosstab = pd.crosstab(df['Dominant_RIASEC'], df['Dominant_Goal'])
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues', ax=ax3)
    ax3.set_title('RIASEC vs Goal Orientation Cross-Tabulation', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Dominant Goal Orientation', fontweight='bold')
    ax3.set_ylabel('Dominant RIASEC Type', fontweight='bold')
    
    # Combined profile scatter
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    x = df['Total_RIASEC']
    y = df['Total_Goal_Orientation']
    
    scatter = ax4.scatter(x, y, c=range(len(df)), cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    
    # Add participant labels
    for i, participant in enumerate(df['participant']):
        ax4.annotate(participant[:8], (x.iloc[i], y.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax4.set_xlabel('Total RIASEC Score', fontweight='bold')
    ax4.set_ylabel('Total Goal Orientation Score', fontweight='bold')
    ax4.set_title('Overall RIASEC vs Goal Orientation Scores', fontweight='bold', fontsize=14)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dominant_types_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Dominant types visualization saved as 'dominant_types_analysis.png'")

def create_mastery_approach_focus_plot(df):
    """Create detailed analysis of Mastery Approach correlations"""
    print("Creating Mastery Approach focus analysis...")
    
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Correlation bar chart for Mastery Approach
    correlations_ma = []
    for dim in riasec_dims:
        corr = np.corrcoef(df[dim], df['Mastery_Approach'])[0, 1]
        correlations_ma.append(corr)
    
    bars = ax1.bar(riasec_dims, correlations_ma, 
                   color=['red' if c < 0 else 'green' for c in correlations_ma],
                   alpha=0.7, edgecolor='black')
    
    # Add correlation values on bars
    for bar, corr in zip(bars, correlations_ma):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height > 0 else height - 0.03,
                f'{corr:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontweight='bold')
    
    ax1.set_title('RIASEC Correlations with Mastery Approach', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Correlation Coefficient', fontweight='bold')
    ax1.set_xlabel('RIASEC Dimensions', fontweight='bold')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.axhline(y=0.3, color='blue', linestyle='--', alpha=0.5, label='Medium Effect (r=0.3)')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Large Effect (r=0.5)')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Detailed scatter plot for strongest correlation (Conventional-Mastery Approach)
    x = df['Conventional']
    y = df['Mastery_Approach']
    
    ax2.scatter(x, y, alpha=0.7, s=100, edgecolors='black', linewidth=1, c='orange')
    
    # Add participant labels
    for i, participant in enumerate(df['participant']):
        ax2.annotate(participant[:8], (x.iloc[i], y.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax2.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
    
    # Calculate and display correlation
    corr = np.corrcoef(x, y)[0, 1]
    ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = 0.008**', transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=12, fontweight='bold')
    
    ax2.set_xlabel('Conventional Interest Score', fontweight='bold')
    ax2.set_ylabel('Mastery Approach Score', fontweight='bold')
    ax2.set_title('Conventional ↔ Mastery Approach\n(Strongest Correlation)', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mastery_approach_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Mastery Approach analysis saved as 'mastery_approach_analysis.png'")

def create_comprehensive_summary_plot(df, correlations):
    """Create a comprehensive summary visualization"""
    print("Creating comprehensive summary plot...")
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Main correlation heatmap (top half)
    ax1 = fig.add_subplot(gs[0:2, 0:3])
    sns.heatmap(correlations, annot=True, cmap='RdBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax1)
    ax1.set_title('RIASEC ↔ Goal Orientation Correlations', fontweight='bold', fontsize=16)
    
    # 2. Distribution summary (top right)
    ax2 = fig.add_subplot(gs[0, 3])
    total_scores = df[['Total_RIASEC', 'Total_Goal_Orientation']]
    total_scores.plot(kind='box', ax=ax2)
    ax2.set_title('Total Scores Distribution', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Dominant types (middle right)
    ax3 = fig.add_subplot(gs[1, 3])
    riasec_counts = df['Dominant_RIASEC'].value_counts()
    ax3.pie(riasec_counts.values, labels=riasec_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Dominant RIASEC Types', fontweight='bold')
    
    # 4. Key findings text (bottom left)
    ax4 = fig.add_subplot(gs[2:4, 0:2])
    ax4.axis('off')
    
    key_findings = [
        "KEY FINDINGS:",
        "",
        "• Strongest correlation: Conventional ↔ Mastery Approach (r = 0.675**)",
        "• Investigative ↔ Mastery Approach also significant (r = 0.534*)",
        "• 71.4% of participants have Investigative as dominant RIASEC type",
        "• 42.9% are primarily Performance Approach oriented",
        "• High achievers: Hardhik, Navya Ennam (high in both approaches)",
        "• Mastery Approach shows positive correlations with all RIASEC types",
        "",
        "INTERPRETATION:",
        "• Learning-oriented motivation connects with career interests",
        "• Conventional interests align strongly with mastery goals",
        "• No strong negative correlations found",
        "• Individual differences show diverse motivation patterns"
    ]
    
    for i, text in enumerate(key_findings):
        weight = 'bold' if text.startswith(('KEY FINDINGS', 'INTERPRETATION', '•')) else 'normal'
        size = 12 if text.startswith(('KEY FINDINGS', 'INTERPRETATION')) else 10
        ax4.text(0.05, 0.95 - i*0.06, text, transform=ax4.transAxes, 
                fontweight=weight, fontsize=size, verticalalignment='top')
    
    # 5. Sample characteristics (bottom right)
    ax5 = fig.add_subplot(gs[2:4, 2:4])
    
    # Create a summary table
    riasec_dims = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    
    means_riasec = df[riasec_dims].mean()
    means_goal = df[goal_dims].mean()
    
    # Plot mean scores
    x_pos = np.arange(len(riasec_dims))
    bars1 = ax5.bar(x_pos - 0.2, means_riasec, 0.4, label='RIASEC', alpha=0.7)
    
    # Normalize goal scores to RIASEC scale for comparison
    means_goal_norm = means_goal / means_goal.max() * means_riasec.max()
    x_pos2 = np.arange(len(goal_dims))
    bars2 = ax5.bar(x_pos2 + len(riasec_dims) + 0.5, means_goal_norm, 0.4, label='Goal Orientation', alpha=0.7)
    
    ax5.set_title('Mean Scores by Dimension', fontweight='bold')
    ax5.set_ylabel('Mean Score', fontweight='bold')
    all_labels = list(riasec_dims) + list(goal_dims)
    ax5.set_xticks(list(x_pos) + list(x_pos2 + len(riasec_dims) + 0.5))
    ax5.set_xticklabels([label[:4] for label in all_labels], rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    plt.suptitle('RIASEC-Goal Orientation Cross-Analysis Summary', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.savefig('comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comprehensive summary saved as 'comprehensive_summary.png'")

def main():
    """Main visualization function"""
    print("RIASEC-GOAL ORIENTATION VISUALIZATION SUITE")
    print("=" * 60)
    
    setup_plot_style()
    df, correlations, p_values = load_data()
    
    # Generate all visualizations
    create_correlation_heatmap(correlations, p_values)
    create_scatter_plots(df)
    create_distribution_plots(df)
    create_participant_profiles_plot(df)
    create_dominant_types_visualization(df)
    create_mastery_approach_focus_plot(df)
    create_comprehensive_summary_plot(df, correlations)
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    visualizations = [
        "correlation_heatmap.png",
        "significant_correlations_scatter.png", 
        "distribution_plots.png",
        "participant_profiles.png",
        "dominant_types_analysis.png",
        "mastery_approach_analysis.png",
        "comprehensive_summary.png"
    ]
    
    for viz in visualizations:
        print(f"- {viz}")
    
    print("\nAll plots are publication-ready with high resolution (300 DPI)")

if __name__ == "__main__":
    main()