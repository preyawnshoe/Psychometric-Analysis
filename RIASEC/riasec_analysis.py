#!/usr/bin/env python3
"""
RIASEC Interest Inventory Analysis
Performs comprehensive descriptive statistical analysis on RIASEC responses.

RIASEC Categories:
- Realistic (R): Hands-on, practical activities
- Investigative (I): Analytical, scientific thinking
- Artistic (A): Creative, expressive activities
- Social (S): Helping, teaching others
- Enterprising (E): Leading, persuading others
- Conventional (C): Organizing, detail-oriented tasks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_and_clean_data(file_path):
    """Load and clean the RIASEC responses data."""
    print("Loading and cleaning data...")
    
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Print basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Number of participants: {len(df)}")
    print(f"Number of questions: {len(df.columns) - 2}")  # Excluding Timestamp and Name columns
    
    return df

def create_riasec_mapping():
    """Create mapping of questions to RIASEC categories based on standard Holland's theory."""
    
    # Question mapping to RIASEC categories (1-based indexing for questions)
    riasec_mapping = {
        # Realistic (R) - hands-on, practical, mechanical
        'R': [1, 7, 14, 22, 30, 32, 37],  # cars, build things, animals, assembling, cook, practical, outdoors
        
        # Investigative (I) - analytical, scientific, problem-solving
        'I': [2, 11, 18, 21, 26, 33, 39],  # puzzles, experiments, science, figure out how things work, analyze, math
        
        # Artistic (A) - creative, expressive, aesthetic
        'A': [3, 8, 17, 23, 27, 31, 41],  # art and music, creative writing, creative person, instruments/sing, acting, draw
        
        # Social (S) - helping, teaching, caring for others
        'S': [4, 12, 13, 20, 28, 34, 40],  # teach/train, help solve problems, healing people, learning about cultures, helping people
        
        # Enterprising (E) - leading, persuading, business-oriented
        'E': [5, 10, 16, 19, 29, 36, 42],  # ambitious/goals, influence/persuade, selling, responsibilities, start business, discussions, lead, speeches
        
        # Conventional (C) - organizing, detail-oriented, systematic
        'C': [6, 9, 15, 24, 25, 35, 38]  # organize things, clear instructions, office work, attention to details, filing/typing, numbers/charts, keeping records, work in office
    }
    
    return riasec_mapping

def convert_responses_to_numeric(df):
    """Convert text responses to numeric values."""
    print("Converting responses to numeric values...")
    
    # Create a copy for processing
    df_numeric = df.copy()
    
    # Define the conversion mapping
    response_mapping = {
        'Yes': 3,
        'May be': 2,
        'No': 1
    }
    
    # Convert all question columns (skip Timestamp and Name)
    question_columns = df.columns[2:]  # Skip first two columns
    
    for col in question_columns:
        df_numeric[col] = df_numeric[col].map(response_mapping)
    
    return df_numeric, question_columns

def calculate_riasec_scores(df_numeric, question_columns, riasec_mapping):
    """Calculate RIASEC scores for each participant."""
    print("Calculating RIASEC scores...")
    
    # Create a DataFrame to store RIASEC scores
    participants = df_numeric['Name Of the Participant: '].values
    riasec_scores = pd.DataFrame(index=participants)
    
    # Calculate scores for each RIASEC category
    for category, question_indices in riasec_mapping.items():
        category_scores = []
        
        for participant_idx in range(len(df_numeric)):
            total_score = 0
            valid_responses = 0
            
            for q_idx in question_indices:
                # Get the question column (adjust for 0-based indexing in column list)
                q_col = question_columns[q_idx - 1]  # Convert to 0-based index
                
                response = df_numeric.iloc[participant_idx][q_col]
                if pd.notna(response):
                    total_score += response
                    valid_responses += 1
            
            # Calculate average score for this category
            if valid_responses > 0:
                avg_score = total_score / valid_responses
            else:
                avg_score = 0
            
            category_scores.append(avg_score)
        
        riasec_scores[category] = category_scores
    
    return riasec_scores

def perform_descriptive_analysis(riasec_scores):
    """Perform comprehensive descriptive statistical analysis."""
    print("\n" + "="*60)
    print("RIASEC DESCRIPTIVE STATISTICAL ANALYSIS")
    print("="*60)
    
    # Basic descriptive statistics
    print("\n1. BASIC DESCRIPTIVE STATISTICS")
    print("-" * 40)
    desc_stats = riasec_scores.describe()
    print(desc_stats.round(3))
    
    # Calculate additional statistics
    print("\n2. ADDITIONAL STATISTICAL MEASURES")
    print("-" * 40)
    
    additional_stats = pd.DataFrame({
        'Variance': riasec_scores.var(),
        'Skewness': riasec_scores.skew(),
        'Kurtosis': riasec_scores.kurtosis()
    })
    print(additional_stats.round(3))
    
    # Frequency analysis of highest scores
    print("\n3. DOMINANT RIASEC TYPE ANALYSIS")
    print("-" * 40)
    
    # Find the dominant type for each participant
    dominant_types = riasec_scores.idxmax(axis=1)
    type_counts = Counter(dominant_types)
    
    print("Frequency of Dominant RIASEC Types:")
    for riasec_type, count in type_counts.most_common():
        percentage = (count / len(riasec_scores)) * 100
        print(f"  {riasec_type}: {count} participants ({percentage:.1f}%)")
    
    # Correlation analysis
    print("\n4. CORRELATION ANALYSIS")
    print("-" * 40)
    correlations = riasec_scores.corr()
    print("Correlation Matrix:")
    print(correlations.round(3))
    
    return desc_stats, additional_stats, type_counts, correlations

def create_visualizations(riasec_scores, desc_stats, type_counts, correlations):
    """Create comprehensive visualizations with improved clarity."""
    print("\nCreating enhanced visualizations...")
    
    # Create multiple figures for better clarity
    plt.rcParams.update({'font.size': 12})
    
    # Figure 1: Distribution Analysis (2x2 grid)
    fig1 = plt.figure(figsize=(16, 12))
    fig1.suptitle('RIASEC Distribution Analysis', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Enhanced Box plot
    plt.subplot(2, 2, 1)
    box_plot = riasec_scores.boxplot(ax=plt.gca(), patch_artist=True, 
                                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                                     medianprops=dict(color='red', linewidth=2))
    plt.title('Score Distribution by RIASEC Category', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Score (1-3 scale)', fontsize=12)
    plt.xlabel('RIASEC Categories', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 3.5)
    
    # 2. Enhanced Bar plot with error bars
    plt.subplot(2, 2, 2)
    means = desc_stats.loc['mean']
    stds = desc_stats.loc['std']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    bars = plt.bar(means.index, means.values, yerr=stds.values, 
                   color=colors, alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    plt.title('Mean RIASEC Scores with Standard Deviation', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Mean Score', fontsize=12)
    plt.xlabel('RIASEC Categories', fontsize=12)
    plt.ylim(0, 3.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value, std in zip(bars, means.values, stds.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.05,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # 3. Enhanced Pie chart
    plt.subplot(2, 2, 3)
    if type_counts:
        colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1'][:len(type_counts)]
        wedges, texts, autotexts = plt.pie(type_counts.values(), labels=type_counts.keys(), 
                                          autopct='%1.1f%%', startangle=90, colors=colors_pie,
                                          explode=[0.05] * len(type_counts))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        plt.title('Distribution of Dominant RIASEC Types', fontsize=14, fontweight='bold', pad=20)
    
    # 4. Violin plot for distribution shape
    plt.subplot(2, 2, 4)
    riasec_data = [riasec_scores[col].values for col in riasec_scores.columns]
    parts = plt.violinplot(riasec_data, positions=range(len(riasec_scores.columns)), 
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    
    plt.xticks(range(len(riasec_scores.columns)), riasec_scores.columns)
    plt.title('Score Distribution Shapes (Violin Plot)', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('Score', fontsize=12)
    plt.xlabel('RIASEC Categories', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('c:/Users/p2var/vandana/RIASEC/riasec_distribution_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Correlation and Profile Analysis
    fig2 = plt.figure(figsize=(16, 10))
    fig2.suptitle('RIASEC Correlation and Individual Profiles', fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Enhanced Correlation heatmap
    plt.subplot(2, 2, 1)
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
    sns.heatmap(correlations, annot=True, cmap='RdYlBu_r', center=0, 
                square=True, fmt='.3f', cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
                linewidths=0.5, mask=mask)
    plt.title('RIASEC Inter-Category Correlations\n(Lower Triangle)', fontsize=14, fontweight='bold')
    
    # 2. All participants radar-style profile
    plt.subplot(2, 2, 2)
    for i, (participant, scores) in enumerate(riasec_scores.iterrows()):
        if i < 8:  # Show first 8 participants for clarity
            plt.plot(scores.index, scores.values, marker='o', linewidth=2, 
                    alpha=0.7, label=f'{participant[:12]}...' if len(participant) > 12 else participant)
    
    plt.title('Individual RIASEC Profiles\n(First 8 Participants)', fontsize=14, fontweight='bold')
    plt.xlabel('RIASEC Categories', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.5, 3.5)
    
    # 3. Score comparison chart
    plt.subplot(2, 2, 3)
    participants_subset = riasec_scores.head(8)  # Top 8 for readability
    
    x = np.arange(len(riasec_scores.columns))
    width = 0.1
    
    for i, (participant, scores) in enumerate(participants_subset.iterrows()):
        offset = (i - len(participants_subset)/2) * width
        plt.bar(x + offset, scores.values, width, 
               label=f'{participant[:10]}...' if len(participant) > 10 else participant,
               alpha=0.8)
    
    plt.title('RIASEC Score Comparison\n(First 8 Participants)', fontsize=14, fontweight='bold')
    plt.xlabel('RIASEC Categories', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.xticks(x, riasec_scores.columns)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 4. Summary statistics visualization
    plt.subplot(2, 2, 4)
    stats_to_plot = desc_stats.loc[['mean', 'std', 'min', 'max']]
    
    x_pos = np.arange(len(riasec_scores.columns))
    width = 0.2
    
    for i, (stat_name, values) in enumerate(stats_to_plot.iterrows()):
        offset = (i - 1.5) * width
        plt.bar(x_pos + offset, values, width, label=stat_name.title(), alpha=0.8)
    
    plt.title('Summary Statistics by Category', fontsize=14, fontweight='bold')
    plt.xlabel('RIASEC Categories', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(x_pos, riasec_scores.columns)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('c:/Users/p2var/vandana/RIASEC/riasec_correlation_profiles.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Figure 3: Individual Analysis
    fig3 = plt.figure(figsize=(20, 12))
    fig3.suptitle('Individual RIASEC Analysis - All Participants', fontsize=18, fontweight='bold', y=0.95)
    
    # Create a grid showing each participant's profile
    n_participants = len(riasec_scores)
    n_cols = 4
    n_rows = (n_participants + n_cols - 1) // n_cols
    
    for i, (participant, scores) in enumerate(riasec_scores.iterrows()):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Create a bar chart for each participant
        colors = ['#FF6B6B' if score == scores.max() else '#4ECDC4' if score >= 2.5 else '#FFA07A' 
                 for score in scores.values]
        
        bars = plt.bar(scores.index, scores.values, color=colors, alpha=0.8)
        plt.title(f'{participant[:15]}...' if len(participant) > 15 else participant, 
                 fontsize=11, fontweight='bold')
        plt.ylim(0, 3.2)
        plt.xticks(rotation=45, fontsize=9)
        
        # Highlight the dominant type
        max_idx = scores.idxmax()
        for j, (cat, bar) in enumerate(zip(scores.index, bars)):
            if cat == max_idx:
                bar.set_edgecolor('red')
                bar.set_linewidth(3)
        
        # Add value labels
        for bar, value in zip(bars, scores.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('c:/Users/p2var/vandana/RIASEC/riasec_individual_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Enhanced visualizations created:")
    print("- riasec_distribution_analysis.png (Distribution analysis)")
    print("- riasec_correlation_profiles.png (Correlations and profiles)")
    print("- riasec_individual_analysis.png (Individual participant analysis)")
    print("- riasec_analysis_plots.png (Original combined plot)")

def generate_detailed_report(riasec_scores, desc_stats, additional_stats, type_counts):
    """Generate a detailed analysis report."""
    print("\n" + "="*60)
    print("DETAILED RIASEC ANALYSIS REPORT")
    print("="*60)
    
    # Sample size
    n_participants = len(riasec_scores)
    print(f"\nSample Size: {n_participants} participants")
    
    # Overall trends
    print("\n5. OVERALL TRENDS AND INSIGHTS")
    print("-" * 40)
    
    # Highest and lowest mean scores
    means = desc_stats.loc['mean']
    highest_category = means.idxmax()
    lowest_category = means.idxmin()
    
    print(f"Highest mean score: {highest_category} ({means[highest_category]:.2f})")
    print(f"Lowest mean score: {lowest_category} ({means[lowest_category]:.2f})")
    
    # Most variable category
    stds = desc_stats.loc['std']
    most_variable = stds.idxmax()
    least_variable = stds.idxmin()
    
    print(f"Most variable category: {most_variable} (SD = {stds[most_variable]:.2f})")
    print(f"Least variable category: {least_variable} (SD = {stds[least_variable]:.2f})")
    
    # Score ranges
    print(f"\nScore Ranges:")
    for category in riasec_scores.columns:
        min_score = riasec_scores[category].min()
        max_score = riasec_scores[category].max()
        print(f"  {category}: {min_score:.2f} - {max_score:.2f}")
    
    # Participant with highest overall scores
    total_scores = riasec_scores.sum(axis=1)
    highest_scorer = total_scores.idxmax()
    print(f"\nHighest overall scorer: {highest_scorer} (Total: {total_scores[highest_scorer]:.2f})")
    
    # Distribution analysis
    print("\n6. DISTRIBUTION CHARACTERISTICS")
    print("-" * 40)
    
    for category in riasec_scores.columns:
        skew = riasec_scores[category].skew()
        kurt = riasec_scores[category].kurtosis()
        
        skew_desc = "right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "approximately symmetric"
        kurt_desc = "heavy-tailed" if kurt > 1 else "light-tailed" if kurt < -1 else "normal-tailed"
        
        print(f"{category}: {skew_desc}, {kurt_desc}")

def save_results_to_csv(riasec_scores, desc_stats):
    """Save analysis results to CSV files."""
    print("\nSaving results to CSV files...")
    
    # Save individual RIASEC scores
    riasec_scores.to_csv('c:/Users/p2var/vandana/RIASEC/riasec_individual_scores.csv')
    print("Individual scores saved to: riasec_individual_scores.csv")
    
    # Save descriptive statistics
    desc_stats.to_csv('c:/Users/p2var/vandana/RIASEC/riasec_descriptive_stats.csv')
    print("Descriptive statistics saved to: riasec_descriptive_stats.csv")

def main():
    """Main analysis function."""
    print("RIASEC Interest Inventory Analysis")
    print("=" * 40)
    
    # File path
    file_path = 'c:/Users/p2var/vandana/RIASEC/responses.csv'
    
    try:
        # Load and clean data
        df = load_and_clean_data(file_path)
        
        # Create RIASEC mapping
        riasec_mapping = create_riasec_mapping()
        
        # Convert responses to numeric
        df_numeric, question_columns = convert_responses_to_numeric(df)
        
        # Calculate RIASEC scores
        riasec_scores = calculate_riasec_scores(df_numeric, question_columns, riasec_mapping)
        
        # Perform descriptive analysis
        desc_stats, additional_stats, type_counts, correlations = perform_descriptive_analysis(riasec_scores)
        
        # Create visualizations
        create_visualizations(riasec_scores, desc_stats, type_counts, correlations)
        
        # Generate detailed report
        generate_detailed_report(riasec_scores, desc_stats, additional_stats, type_counts)
        
        # Save results
        save_results_to_csv(riasec_scores, desc_stats)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Files generated:")
        print("- riasec_analysis_plots.png (visualizations)")
        print("- riasec_individual_scores.csv (individual scores)")
        print("- riasec_descriptive_stats.csv (summary statistics)")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()