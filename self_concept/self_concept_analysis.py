import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load and process the data
def load_and_process_data():
    """Load the CSV data and process it for analysis."""
    df = pd.read_csv('responses.csv')
    
    # Extract the columns that contain the scale items (questions 1-49)
    scale_columns = []
    for col in df.columns:
        if any(str(i) + '.' in col for i in range(1, 50)):
            scale_columns.append(col)
    
    print(f"Found {len(scale_columns)} scale items")
    print("\nParticipants in the study:")
    for i, name in enumerate(df['Name'], 1):
        print(f"{i}. {name}")
    
    return df, scale_columns

def convert_likert_to_numeric(response):
    """Convert Likert scale responses to numeric values."""
    if pd.isna(response):
        return np.nan
    
    response_str = str(response).lower().strip()
    
    # Map responses to numeric scale (1-5, higher = more positive self-concept)
    mapping = {
        'strongly disagree': 1,
        'strongly_disagree': 1,
        'disagree': 2,
        'undecided': 3,
        'agree': 4,
        'strongly agree': 5,
        'strongly_agree': 5
    }
    
    return mapping.get(response_str, 3)  # Default to 3 (undecided) for unclear responses

def identify_reverse_scored_items():
    """Identify items that should be reverse scored based on their content."""
    # Items that indicate negative self-concept (should be reverse scored)
    reverse_items = [3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 27, 28, 29, 30, 32, 36, 37, 39, 43, 48]
    return reverse_items

def calculate_self_concept_scores(df, scale_columns):
    """Calculate self-concept scores for each participant."""
    reverse_items = identify_reverse_scored_items()
    
    # Convert all responses to numeric
    numeric_data = df[scale_columns].copy()
    
    for col in scale_columns:
        numeric_data[col] = df[col].apply(convert_likert_to_numeric)
    
    # Reverse score appropriate items
    for i, col in enumerate(scale_columns, 1):
        if i in reverse_items:
            # Reverse score: 1->5, 2->4, 3->3, 4->2, 5->1
            numeric_data[col] = 6 - numeric_data[col]
    
    # Calculate total scores (sum of all items)
    total_scores = numeric_data.sum(axis=1)
    
    # Calculate mean scores (average across items)
    mean_scores = numeric_data.mean(axis=1)
    
    # Create results dataframe
    results = pd.DataFrame({
        'Name': df['Name'],
        'Age': df['Age'],
        'Gender': df['Gender'],
        'Total_Score': total_scores,
        'Mean_Score': mean_scores,
        'Valid_Responses': numeric_data.count(axis=1)
    })
    
    return results, numeric_data

def calculate_percentiles(results):
    """Calculate percentile rankings for participants."""
    results['Percentile'] = stats.rankdata(results['Total_Score'], method='average') / len(results) * 100
    results['Percentile_Rank'] = results['Total_Score'].rank(pct=True) * 100
    
    # Sort by total score (descending)
    results_sorted = results.sort_values('Total_Score', ascending=False).reset_index(drop=True)
    results_sorted['Rank'] = range(1, len(results_sorted) + 1)
    
    return results_sorted

def create_visualizations(results):
    """Create distribution plots and visualizations."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Self-Concept Scale Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram of total scores
    axes[0, 0].hist(results['Total_Score'], bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Total Self-Concept Scores')
    axes[0, 0].set_xlabel('Total Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add statistics to the plot
    mean_score = results['Total_Score'].mean()
    std_score = results['Total_Score'].std()
    axes[0, 0].axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.1f}')
    axes[0, 0].legend()
    
    # 2. Box plot of scores by gender
    if results['Gender'].nunique() > 1:
        sns.boxplot(data=results, x='Gender', y='Total_Score', ax=axes[0, 1])
        axes[0, 1].set_title('Self-Concept Scores by Gender')
        axes[0, 1].set_ylabel('Total Score')
    else:
        axes[0, 1].text(0.5, 0.5, 'Insufficient gender diversity\nfor comparison', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Self-Concept Scores by Gender')
    
    # 3. Q-Q plot to check normality
    stats.probplot(results['Total_Score'], dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot (Normality Check)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Percentile rankings bar chart
    top_10 = results.nlargest(10, 'Total_Score')
    bars = axes[1, 1].bar(range(len(top_10)), top_10['Total_Score'], 
                         color='lightcoral', alpha=0.7)
    axes[1, 1].set_title('Top Participants by Self-Concept Score')
    axes[1, 1].set_xlabel('Participant Rank')
    axes[1, 1].set_ylabel('Total Score')
    axes[1, 1].set_xticks(range(len(top_10)))
    axes[1, 1].set_xticklabels([f"{i+1}" for i in range(len(top_10))])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add participant names as labels on bars
    for i, (bar, name) in enumerate(zip(bars, top_10['Name'])):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       name.split()[0], ha='center', va='bottom', rotation=45, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('self_concept_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_summary_report(results):
    """Generate a comprehensive summary report."""
    print("\n" + "="*80)
    print("SELF-CONCEPT SCALE ANALYSIS REPORT")
    print("="*80)
    
    # Basic statistics
    print(f"\nSAMPLE CHARACTERISTICS:")
    print(f"Total participants: {len(results)}")
    print(f"Age range: {results['Age'].min()}-{results['Age'].max()} years")
    print(f"Mean age: {results['Age'].mean():.1f} years")
    print(f"Gender distribution: {results['Gender'].value_counts().to_dict()}")
    
    # Score statistics
    print(f"\nSCORE STATISTICS:")
    print(f"Mean total score: {results['Total_Score'].mean():.2f}")
    print(f"Standard deviation: {results['Total_Score'].std():.2f}")
    print(f"Minimum score: {results['Total_Score'].min()}")
    print(f"Maximum score: {results['Total_Score'].max()}")
    print(f"Score range: {results['Total_Score'].max() - results['Total_Score'].min()}")
    
    # Percentile information
    print(f"\nPERCENTILE BREAKDOWNS:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        score = np.percentile(results['Total_Score'], p)
        print(f"{p}th percentile: {score:.1f}")
    
    # Top performers
    print(f"\nTOP 5 PARTICIPANTS (Highest Self-Concept):")
    top_5 = results.nlargest(5, 'Total_Score')
    for i, row in top_5.iterrows():
        print(f"{row['Rank']}. {row['Name']} - Score: {row['Total_Score']:.0f} "
              f"(Percentile: {row['Percentile_Rank']:.1f}%)")
    
    # Bottom performers
    print(f"\nBOTTOM 5 PARTICIPANTS (Lowest Self-Concept):")
    bottom_5 = results.nsmallest(5, 'Total_Score')
    for i, row in bottom_5.iterrows():
        rank = len(results) - list(results.sort_values('Total_Score', ascending=False).index).index(i)
        print(f"{rank}. {row['Name']} - Score: {row['Total_Score']:.0f} "
              f"(Percentile: {row['Percentile_Rank']:.1f}%)")
    
    # Complete ranking table
    print(f"\nCOMPLETE RANKING TABLE:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Name':<20} {'Age':<4} {'Gender':<8} {'Score':<6} {'Percentile':<10}")
    print("-" * 80)
    for _, row in results.iterrows():
        print(f"{row['Rank']:<5} {row['Name']:<20} {row['Age']:<4} {row['Gender']:<8} "
              f"{row['Total_Score']:<6.0f} {row['Percentile_Rank']:<10.1f}%")
    
    return results

def main():
    """Main analysis function."""
    print("Self-Concept Scale Analysis")
    print("="*40)
    
    # Load and process data
    df, scale_columns = load_and_process_data()
    
    # Calculate scores
    results, numeric_data = calculate_self_concept_scores(df, scale_columns)
    
    # Calculate percentiles and rankings
    results = calculate_percentiles(results)
    
    # Create visualizations
    create_visualizations(results)
    
    # Generate summary report
    final_results = generate_summary_report(results)
    
    # Save results to CSV
    final_results.to_csv('self_concept_results.csv', index=False)
    print(f"\n\nResults saved to 'self_concept_results.csv'")
    print("Visualization saved to 'self_concept_analysis.png'")
    
    return final_results

if __name__ == "__main__":
    results = main()