"""
Fixed Comprehensive RIASEC Career Analysis
Handles the actual structure of the careers_map.csv file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare the RIASEC responses and career mapping data"""
    print("📊 Loading RIASEC data...")
    
    # Load RIASEC responses
    riasec_responses = pd.read_csv('responses_72.csv')
    print(f"✓ Loaded {len(riasec_responses)} participant responses")
    
    # Load career mapping data with exact RIASEC values
    careers_df = pd.read_csv('career.csv')
    print(f"✓ Loaded career data with {len(careers_df)} careers")
    print(f"✓ Found {careers_df['Domain'].nunique()} domains")
    
    return riasec_responses, careers_df

def convert_riasec_responses(responses_df):
    """Convert RIASEC responses to numerical scores"""
    print("🔄 Converting participant responses to RIASEC scores...")
    
    # Extended RIASEC question mapping (72 questions, 12 per type)
    riasec_mapping = {
        'Realistic': [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67],
        'Investigative': [2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68],
        'Artistic': [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69],
        'Social': [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70],
        'Enterprising': [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71],
        'Conventional': [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    }
    
    # Response value mapping (handle variations in responses)
    response_mapping = {'Yes': 2, 'Maybe': 1, 'May be': 1, 'No': 0}
    
    participants_riasec = []
    
    for _, row in responses_df.iterrows():
        # Get participant name (second column)
        participant_name = str(row.iloc[1]).strip()
        
        # Get responses (starting from column 7, skipping metadata columns)
        all_responses = row.iloc[7:].values  # Skip first 7 columns (Timestamp, Name, Gender, Contact, Course, College, Location)
        
        # Calculate raw RIASEC scores
        raw_riasec_scores = {}
        
        for riasec_type, question_indices in riasec_mapping.items():
            type_responses = []
            
            for q_idx in question_indices:
                if q_idx-1 < len(all_responses):  # Adjust for 0-based indexing
                    response = all_responses[q_idx-1]
                    if pd.notna(response) and str(response).strip() in response_mapping:
                        type_responses.append(response_mapping[str(response).strip()])
                    else:
                        type_responses.append(0)  # Default to 'No' if missing
            
            # Calculate raw average score for this RIASEC type
            raw_riasec_scores[riasec_type] = np.mean(type_responses) if type_responses else 0.0
        
        # Calculate RIASEC scores by simply using the average of responses for each type
        riasec_scores = {'Participant': participant_name}
        riasec_types = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
        
        # Use the raw average scores directly (no normalization or top-3 selection)
        for riasec_type in riasec_types:
            riasec_scores[riasec_type] = raw_riasec_scores[riasec_type]
        
        # Calculate total score and Holland code
        riasec_values = [riasec_scores[dim] for dim in riasec_types]
        riasec_scores['Total_RIASEC'] = np.sum(riasec_values)
        
        # Generate Holland code (top 3 types with highest scores)
        all_types = [(dim, riasec_scores[dim]) for dim in riasec_types]
        all_types.sort(key=lambda x: x[1], reverse=True)
        
        riasec_scores['Holland_Code'] = ''.join([t[0][0] for t in all_types[:3]])
        riasec_scores['Dominant_Type'] = all_types[0][0] if all_types else 'None'
        riasec_scores['Secondary_Type'] = all_types[1][0] if len(all_types) > 1 else 'None'
        riasec_scores['Tertiary_Type'] = all_types[2][0] if len(all_types) > 2 else 'None'
        
        participants_riasec.append(riasec_scores)
    
    participant_df = pd.DataFrame(participants_riasec)
    print(f"✓ Processed {len(participant_df)} participants")
    
    return participant_df

def create_career_riasec_mappings(careers_df):
    """Use exact RIASEC values from career.csv"""
    print("🏢 Using exact RIASEC values from career.csv...")
    
    careers_with_riasec = []
    
    for _, career_row in careers_df.iterrows():
        domain = career_row['Domain']
        career = career_row['Career']
        
        # Extract exact RIASEC values from the CSV
        riasec_profile = {
            'Realistic': career_row['R'],
            'Investigative': career_row['I'], 
            'Artistic': career_row['A'],
            'Social': career_row['S'],
            'Enterprising': career_row['E'],
            'Conventional': career_row['C']
        }
        
        # Calculate interest score based on actual RIASEC values
        interest_score = np.mean(list(riasec_profile.values()))
        
        career_data = {
            'Domain': domain,
            'Career': career,
            'Interest_Score': interest_score,
            **riasec_profile
        }
        
        careers_with_riasec.append(career_data)
    
    result_df = pd.DataFrame(careers_with_riasec)
    print(f"✓ Using exact RIASEC values for {len(result_df)} careers")
    
    return result_df

def calculate_career_matches(participant_riasec, career_data):
    """Calculate career matches using simple multiplication of RIASEC scores"""
    print("🎯 Calculating career matches using multiplication...")

    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']

    all_matches = []
    total_comparisons = len(participant_riasec) * len(career_data)

    for _, participant in participant_riasec.iterrows():
        participant_scores = np.array([participant[col] for col in riasec_cols])

        for _, career in career_data.iterrows():
            career_scores = np.array([career[col] for col in riasec_cols])

            # Simple multiplication approach
            # Multiply corresponding RIASEC scores and sum the results
            multiplied_scores = participant_scores * career_scores
            combined_score = np.sum(multiplied_scores)

            # Normalize by maximum possible score (3 * max career score)
            # Since participant scores sum to 3, and career scores max around 2.5
            max_possible = 3 * np.max(career_scores)
            if max_possible > 0:
                normalized_score = combined_score / max_possible
            else:
                normalized_score = 0

            match_data = {
                'Participant': participant['Participant'],
                'Career': career['Career'],
                'Domain': career['Domain'],
                'Interest_Score': career['Interest_Score'],
                'Multiplied_Score': combined_score,
                'Combined_Score': normalized_score,  # Use normalized score for ranking
                'Holland_Code': participant['Holland_Code'],
                'Dominant_Type': participant['Dominant_Type']
            }

            # Add individual RIASEC comparisons
            for i, riasec_type in enumerate(riasec_cols):
                match_data[f'{riasec_type}_Participant'] = participant_scores[i]
                match_data[f'{riasec_type}_Career'] = career_scores[i]
                match_data[f'{riasec_type}_Product'] = multiplied_scores[i]

            all_matches.append(match_data)

    print(f"✓ Calculated {len(all_matches)} total matches using multiplication")
    return pd.DataFrame(all_matches)

def generate_recommendations(matches_df):
    """Generate top 5 careers from top 4 domains for each participant"""
    print("🏆 Generating top 5 careers from top 4 domains...")

    recommendations = {}

    for participant in matches_df['Participant'].unique():
        participant_matches = matches_df[matches_df['Participant'] == participant]

        # First, identify the top 4 domains by getting highest scoring career from each domain
        domain_best_scores = participant_matches.groupby('Domain')['Combined_Score'].max().sort_values(ascending=False)
        top_4_domains = domain_best_scores.head(4).index.tolist()
        
        # Get top 5 careers from each of the top 4 domains
        domain_recommendations = []
        
        for domain in top_4_domains:
            domain_careers = participant_matches[participant_matches['Domain'] == domain]
            top_5_in_domain = domain_careers.nlargest(5, 'Combined_Score')
            
            for _, career in top_5_in_domain.iterrows():
                domain_recommendations.append(career)
        
        # Convert to DataFrame and sort by score (total of up to 20 careers)
        all_recommendations = pd.DataFrame(domain_recommendations).sort_values('Combined_Score', ascending=False)
        
        # Take top 20 overall (5 from each of 4 domains)
        top_careers = all_recommendations.head(20)

        # Calculate statistics
        avg_score = top_careers['Combined_Score'].mean()
        top_domains = top_careers['Domain'].value_counts().to_dict()

        recommendations[participant] = {
            'top_careers': top_careers,
            'avg_score': avg_score,
            'top_domains': top_domains,
            'holland_code': top_careers['Holland_Code'].iloc[0],
            'dominant_type': top_careers['Dominant_Type'].iloc[0],
            'total_careers': len(top_careers),
            'top_4_domains': top_4_domains
        }

    return recommendations

def create_visualizations(participant_riasec, career_data, recommendations):
    """Create comprehensive visualizations"""
    print("📊 Creating visualizations...")
    
    os.makedirs('career_analysis_output', exist_ok=True)
    
    # 1. Individual RIASEC Profiles
    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    n_participants = len(participant_riasec)
    n_cols = 4
    n_rows = (n_participants + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (_, participant) in enumerate(participant_riasec.iterrows()):
        if i < len(axes):
            ax = axes[i]
            scores = [participant[col] for col in riasec_cols]
            
            bars = ax.bar(range(len(riasec_cols)), scores, color=colors)
            ax.set_title(f'{participant["Participant"]}\nCode: {participant["Holland_Code"]}', 
                        fontsize=10, fontweight='bold')
            ax.set_xticks(range(len(riasec_cols)))
            ax.set_xticklabels([col[0] for col in riasec_cols])
            ax.set_ylim(0, 3.2)
            ax.grid(True, alpha=0.3)
            
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                       f'{score:.2f}', ha='center', va='bottom', fontsize=8)
    
    for i in range(len(participant_riasec), len(axes)):
        axes[i].remove()
    
    plt.suptitle('Individual RIASEC Profiles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('career_analysis_output/individual_riasec_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Domain Analysis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Career count by domain
    domain_counts = career_data['Domain'].value_counts().head(10)
    domain_counts.plot(kind='barh', ax=ax1, color='skyblue')
    ax1.set_title('Career Distribution by Domain', fontweight='bold')
    
    # RIASEC type distribution among participants
    type_dist = participant_riasec['Dominant_Type'].value_counts()
    ax2.pie(type_dist.values, labels=type_dist.index, autopct='%1.1f%%', 
           colors=colors[:len(type_dist)])
    ax2.set_title('Participant RIASEC Type Distribution', fontweight='bold')
    
    # Average RIASEC scores
    avg_riasec = participant_riasec[riasec_cols].mean()
    avg_riasec.plot(kind='bar', ax=ax3, color=colors)
    ax3.set_title('Average RIASEC Scores - Participants', fontweight='bold')
    ax3.set_ylabel('Average Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # Recommendation quality
    all_scores = []
    for participant, data in recommendations.items():
        all_scores.extend(data['top_careers']['Combined_Score'].tolist())
    
    ax4.hist(all_scores, bins=15, color='lightcoral', alpha=0.7)
    ax4.set_title('Distribution of Top Recommendation Scores', fontweight='bold')
    ax4.set_xlabel('Match Score')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('career_analysis_output/analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Top Recommendations Summary
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Most recommended careers
    all_careers = []
    for participant, data in recommendations.items():
        all_careers.extend([career['Career'] for _, career in data['top_careers'].iterrows()])
    
    from collections import Counter
    top_careers = Counter(all_careers).most_common(15)
    careers, counts = zip(*top_careers)
    
    ax1.barh(range(len(careers)), counts, color='lightgreen')
    ax1.set_yticks(range(len(careers)))
    ax1.set_yticklabels(careers, fontsize=9)
    ax1.set_title('Most Frequently Recommended Careers', fontweight='bold')
    ax1.set_xlabel('Number of Recommendations')
    
    # Most recommended domains
    all_domains = []
    for participant, data in recommendations.items():
        all_domains.extend([career['Domain'] for _, career in data['top_careers'].iterrows()])
    
    top_domains = Counter(all_domains).most_common(10)
    domains, counts = zip(*top_domains)
    
    ax2.bar(range(len(domains)), counts, color='lightsalmon')
    ax2.set_xticks(range(len(domains)))
    ax2.set_xticklabels(domains, rotation=45, ha='right')
    ax2.set_title('Most Frequently Recommended Domains', fontweight='bold')
    ax2.set_ylabel('Number of Recommendations')
    
    plt.tight_layout()
    plt.savefig('career_analysis_output/top_recommendations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ All visualizations created!")

def generate_reports(participant_riasec, career_data, recommendations, matches_df):
    """Generate comprehensive reports"""
    print("📝 Generating reports...")
    
    # Summary CSV
    summary_data = []
    detailed_data = []
    
    for participant, rec_data in recommendations.items():
        top_careers = rec_data['top_careers']
        
        # Summary row
        career_list = [career['Career'] for _, career in top_careers.iterrows()]
        domain_list = [career['Domain'] for _, career in top_careers.iterrows()]

        # Create structured summary for top 3 domains
        summary_row = {
            'Participant': participant,
            'Holland_Code': rec_data['holland_code'],
            'Dominant_Type': rec_data['dominant_type'],
            'Avg_Match_Score': rec_data['avg_score'],
            'Total_Careers': rec_data['total_careers']
        }

        # Add top 4 domains with their careers
        top_4_domains = rec_data['top_4_domains']
        for domain_idx, domain in enumerate(top_4_domains, 1):
            summary_row[f'Domain_{domain_idx}'] = domain
            
            # Get careers for this domain
            domain_careers = [career for _, career in top_careers.iterrows() if career['Domain'] == domain]
            
            # Add up to 5 careers for this domain
            for career_idx, career in enumerate(domain_careers[:5], 1):
                summary_row[f'Domain_{domain_idx}_Career_{career_idx}'] = career['Career']
                summary_row[f'Domain_{domain_idx}_Score_{career_idx}'] = career['Combined_Score']

        summary_data.append(summary_row)
        
        # Detailed rows
        for i, (_, career) in enumerate(top_careers.iterrows()):
            detailed_row = {
                'Participant': participant,
                'Rank': i + 1,
                'Career': career['Career'],
                'Domain': career['Domain'],
                'Combined_Score': career['Combined_Score'],
                'Interest_Score': career['Interest_Score'],
                'Holland_Code': career['Holland_Code']
            }
            detailed_data.append(detailed_row)
    
    # Save CSV files
    pd.DataFrame(summary_data).to_csv('career_analysis_output/career_recommendations_summary.csv', index=False)
    pd.DataFrame(detailed_data).to_csv('career_analysis_output/career_recommendations_detailed.csv', index=False)
    
    # Text report
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE RIASEC CAREER ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 50)
    report.append(f"• Participants Analyzed: {len(participant_riasec)}")
    report.append(f"• Careers in Database: {len(career_data)}")
    report.append(f"• Career Domains: {career_data['Domain'].nunique()}")
    report.append(f"• Total Matches Calculated: {len(matches_df)}")
    report.append("")
    
    # Individual recommendations
    report.append("INDIVIDUAL RECOMMENDATIONS")
    report.append("-" * 50)
    
    for participant, rec_data in recommendations.items():
        report.append(f"\n🎯 {participant.upper()}")
        report.append("-" * 30)
        
        participant_data = participant_riasec[participant_riasec['Participant'] == participant].iloc[0]
        riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
        
        report.append(f"Holland Code: {rec_data['holland_code']}")
        report.append(f"Dominant Type: {rec_data['dominant_type']}")
        report.append(f"RIASEC Profile: " + " | ".join([f"{col[0]}:{participant_data[col]:.2f}" for col in riasec_cols]))
        report.append(f"Average Match Score: {rec_data['avg_score']:.3f}")
        report.append(f"Total Careers Recommended: {rec_data['total_careers']}")
        report.append("")

        report.append(f"Top 4 Domains: {', '.join(rec_data['top_4_domains'])}")
        report.append("")
        report.append("Top 5 Careers from Each of Top 4 Domains:")
        
        current_domain = ""
        domain_count = {}
        
        for i, (_, career) in enumerate(rec_data['top_careers'].iterrows()):
            career_domain = career['Domain']
            
            # Track careers per domain
            if career_domain not in domain_count:
                domain_count[career_domain] = 0
            domain_count[career_domain] += 1
            
            # Show domain header for new domains
            if career_domain != current_domain:
                if i > 0:  # Add space before new domain (except first)
                    report.append("")
                report.append(f"  📁 {career_domain} Domain:")
                current_domain = career_domain
            
            report.append(f"    {domain_count[career_domain]}. {career['Career']}")
            report.append(f"       Match Score: {career['Combined_Score']:.3f}")
            report.append(f"       Raw Product Score: {career['Multiplied_Score']:.3f}")
            
            if i < len(rec_data['top_careers']) - 1:  # Add space between entries except last
                report.append("")
    
    # Save text report
    with open('career_analysis_output/Comprehensive_Career_Analysis_Report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✓ Reports generated!")

def main():
    """Main analysis function"""
    print("🚀 COMPREHENSIVE RIASEC CAREER ANALYSIS")
    print("=" * 60)
    
    try:
        # Load data
        riasec_responses, careers_df = load_and_prepare_data()
        
        # Convert responses
        participant_riasec = convert_riasec_responses(riasec_responses)
        
        # Create career RIASEC mappings
        career_data = create_career_riasec_mappings(careers_df)
        
        # Calculate matches
        matches_df = calculate_career_matches(participant_riasec, career_data)
        
        # Generate recommendations
        recommendations = generate_recommendations(matches_df)
        
        # Create visualizations
        create_visualizations(participant_riasec, career_data, recommendations)
        
        # Generate reports
        generate_reports(participant_riasec, career_data, recommendations, matches_df)
        
        # Save data files
        participant_riasec.to_csv('career_analysis_output/participant_riasec_scores.csv', index=False)
        career_data.to_csv('career_analysis_output/career_riasec_profiles.csv', index=False)
        matches_df.to_csv('career_analysis_output/all_career_matches.csv', index=False)
        
        print("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Generated Files:")
        print("  📊 Visualizations:")
        print("    • individual_riasec_profiles.png")
        print("    • analysis_overview.png")
        print("    • top_recommendations.png")
        print("  📄 Reports:")
        print("    • career_recommendations_summary.csv (top 5 from top 4 domains)")
        print("    • career_recommendations_detailed.csv (ranked recommendations)")
        print("    • Comprehensive_Career_Analysis_Report.txt")
        print("  📋 Data Files:")
        print("    • participant_riasec_scores.csv")
        print("    • career_riasec_profiles.csv")
        print("    • all_career_matches.csv")
        
        # Summary statistics
        avg_match_score = np.mean([r['avg_score'] for r in recommendations.values()])
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"  • Participants: {len(participant_riasec)}")
        print(f"  • Careers: {len(career_data)}")
        print(f"  • Domains: {career_data['Domain'].nunique()}")
        print(f"  • Average Match Quality: {avg_match_score:.3f}")
        
        # Sample result
        sample_participant = list(recommendations.keys())[0]
        sample_career = recommendations[sample_participant]['top_careers'].iloc[0]
        print(f"\n🎯 Sample Result:")
        print(f"  {sample_participant} → {sample_career['Career']}")
        print(f"  Match Score: {sample_career['Combined_Score']:.3f}")
        
        return recommendations, participant_riasec, career_data
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()