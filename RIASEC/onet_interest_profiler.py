"""
O*NET Interest Profiler Career Analysis
Based on the official O*NET Interest Profiler methodology
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the RIASEC responses and career mapping data"""
    print("📊 Loading O*NET Interest Profiler data...")
    
    # Load RIASEC responses
    riasec_responses = pd.read_csv('responses_72.csv')
    print(f"✓ Loaded {len(riasec_responses)} participant responses")
    
    # Load career mapping data with exact RIASEC values
    careers_df = pd.read_csv('career.csv')
    print(f"✓ Loaded career data with {len(careers_df)} careers")
    print(f"✓ Found {careers_df['Domain'].nunique()} domains")
    
    return riasec_responses, careers_df

def convert_onet_interest_scores(responses_df):
    """Convert responses using O*NET Interest Profiler methodology"""
    print("🔄 Converting responses using O*NET Interest Profiler method...")
    
    # O*NET Interest Profiler question mapping (72 questions distributed across RIASEC)
    onet_mapping = {
        'Realistic': [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67],
        'Investigative': [2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68],
        'Artistic': [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69],
        'Social': [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70],
        'Enterprising': [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71],
        'Conventional': [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    }
    
    # O*NET Interest Profiler scoring: Like Very Much=3, Like=2, Dislike=1, Dislike Very Much=0
    # We'll map our responses: Yes=3, Maybe=2, No=1 (adjusted for our format)
    onet_response_mapping = {'Yes': 3, 'Maybe': 2, 'May be': 2, 'No': 1}
    
    participants_onet = []
    
    for _, row in responses_df.iterrows():
        # Get participant name
        participant_name = str(row.iloc[1]).strip()
        
        # Get responses (starting from column 7, skipping metadata columns)
        all_responses = row.iloc[7:].values
        
        # Calculate O*NET Interest scores
        onet_scores = {}
        raw_totals = {}
        
        for interest_type, question_indices in onet_mapping.items():
            type_responses = []
            
            for q_idx in question_indices:
                if q_idx-1 < len(all_responses):
                    response = all_responses[q_idx-1]
                    if pd.notna(response) and str(response).strip() in onet_response_mapping:
                        type_responses.append(onet_response_mapping[str(response).strip()])
                    else:
                        type_responses.append(1)  # Default to 'Dislike' if missing
            
            # Calculate raw total score (sum, not average like standard RIASEC)
            raw_total = sum(type_responses)
            raw_totals[interest_type] = raw_total
        
        # O*NET Interest Level calculation
        # Convert raw scores to interest levels: High (23-36), Moderate (15-22), Low (12-14), Very Low (0-11)
        interest_levels = {}
        for interest_type, raw_score in raw_totals.items():
            if raw_score >= 23:
                interest_levels[interest_type] = 'High'
            elif raw_score >= 15:
                interest_levels[interest_type] = 'Moderate'
            elif raw_score >= 12:
                interest_levels[interest_type] = 'Low'
            else:
                interest_levels[interest_type] = 'Very Low'
        
        # For numerical matching, normalize scores to 0-100 scale (O*NET standard)
        max_possible = 12 * 3  # 12 questions * 3 max score
        for interest_type, raw_score in raw_totals.items():
            onet_scores[interest_type] = (raw_score / max_possible) * 100
        
        # Create participant profile
        participant_profile = {
            'Participant': participant_name,
            'Realistic': onet_scores['Realistic'],
            'Investigative': onet_scores['Investigative'], 
            'Artistic': onet_scores['Artistic'],
            'Social': onet_scores['Social'],
            'Enterprising': onet_scores['Enterprising'],
            'Conventional': onet_scores['Conventional']
        }
        
        # Add raw totals for reference
        for interest_type in onet_mapping.keys():
            participant_profile[f'{interest_type}_Raw'] = raw_totals[interest_type]
            participant_profile[f'{interest_type}_Level'] = interest_levels[interest_type]
        
        # Calculate total interest score
        riasec_types = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
        participant_profile['Total_Interest_Score'] = sum(onet_scores[t] for t in riasec_types)
        
        # Generate Holland code based on top 3 highest scores
        type_scores = [(t, onet_scores[t]) for t in riasec_types]
        type_scores.sort(key=lambda x: x[1], reverse=True)
        
        participant_profile['Holland_Code'] = ''.join([t[0][0] for t in type_scores[:3]])
        participant_profile['Primary_Interest'] = type_scores[0][0]
        participant_profile['Secondary_Interest'] = type_scores[1][0] if len(type_scores) > 1 else 'None'
        participant_profile['Tertiary_Interest'] = type_scores[2][0] if len(type_scores) > 2 else 'None'
        
        # Add interest level summary
        high_interests = [k for k, v in interest_levels.items() if v == 'High']
        moderate_interests = [k for k, v in interest_levels.items() if v == 'Moderate']
        
        participant_profile['High_Interest_Areas'] = ', '.join(high_interests) if high_interests else 'None'
        participant_profile['Moderate_Interest_Areas'] = ', '.join(moderate_interests) if moderate_interests else 'None'
        participant_profile['Interest_Pattern'] = f"{len(high_interests)}H-{len(moderate_interests)}M"
        
        participants_onet.append(participant_profile)
    
    participant_df = pd.DataFrame(participants_onet)
    print(f"✓ Processed {len(participant_df)} participants using O*NET methodology")
    
    return participant_df

def calculate_onet_career_matches(participant_onet, career_data):
    """Calculate career matches using O*NET Interest Profiler correlation method"""
    print("🎯 Calculating career matches using O*NET correlation method...")
    
    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    all_matches = []
    
    for _, participant in participant_onet.iterrows():
        participant_scores = np.array([participant[col] for col in riasec_cols])
        
        for _, career in career_data.iterrows():
            # Convert career RIASEC values to 0-100 scale to match O*NET format
            career_scores = np.array([career[col] * 20 for col in ['R', 'I', 'A', 'S', 'E', 'C']])  # Scale up from 0-5 to 0-100
            
            # O*NET uses Pearson correlation coefficient for matching
            if np.std(participant_scores) > 0 and np.std(career_scores) > 0:
                correlation = np.corrcoef(participant_scores, career_scores)[0, 1]
                # Convert correlation (-1 to 1) to match score (0 to 100)
                match_score = ((correlation + 1) / 2) * 100
            else:
                match_score = 50  # Neutral score if no variation
            
            # Calculate additional O*NET metrics
            # Euclidean distance (lower is better)
            euclidean_distance = np.sqrt(np.sum((participant_scores - career_scores) ** 2))
            
            # Profile similarity (dot product normalized)
            dot_product = np.dot(participant_scores, career_scores)
            magnitude_p = np.sqrt(np.sum(participant_scores ** 2))
            magnitude_c = np.sqrt(np.sum(career_scores ** 2))
            
            if magnitude_p > 0 and magnitude_c > 0:
                cosine_similarity = (dot_product / (magnitude_p * magnitude_c)) * 100
            else:
                cosine_similarity = 50
            
            # Combined O*NET match score (weighted average)
            onet_combined_score = (match_score * 0.5) + (cosine_similarity * 0.3) + ((100 - min(euclidean_distance, 100)) * 0.2)
            
            match_data = {
                'Participant': participant['Participant'],
                'Career': career['Career'],
                'Domain': career['Domain'],
                'ONET_Match_Score': onet_combined_score,
                'Correlation_Score': match_score,
                'Cosine_Similarity': cosine_similarity,
                'Euclidean_Distance': euclidean_distance,
                'Holland_Code': participant['Holland_Code'],
                'Primary_Interest': participant['Primary_Interest'],
                'Interest_Pattern': participant['Interest_Pattern']
            }
            
            # Add individual interest area comparisons
            for i, interest_type in enumerate(riasec_cols):
                match_data[f'{interest_type}_Participant'] = participant_scores[i]
                match_data[f'{interest_type}_Career'] = career_scores[i]
                match_data[f'{interest_type}_Diff'] = abs(participant_scores[i] - career_scores[i])
            
            all_matches.append(match_data)
    
    print(f"✓ Calculated {len(all_matches)} career matches using O*NET methodology")
    return pd.DataFrame(all_matches)

def generate_onet_recommendations(matches_df):
    """Generate career recommendations using O*NET Interest Profiler approach"""
    print("🏆 Generating O*NET Interest Profiler recommendations...")
    
    recommendations = {}
    
    for participant in matches_df['Participant'].unique():
        participant_matches = matches_df[matches_df['Participant'] == participant]
        
        # O*NET approach: Top careers by match score, organized by interest level
        top_matches = participant_matches.nlargest(20, 'ONET_Match_Score')
        
        # Categorize matches by score ranges (O*NET style)
        excellent_matches = top_matches[top_matches['ONET_Match_Score'] >= 80]
        good_matches = top_matches[(top_matches['ONET_Match_Score'] >= 65) & (top_matches['ONET_Match_Score'] < 80)]
        fair_matches = top_matches[(top_matches['ONET_Match_Score'] >= 50) & (top_matches['ONET_Match_Score'] < 65)]
        
        # Domain diversity analysis
        top_domains = top_matches['Domain'].value_counts().head(5).to_dict()
        
        # Calculate statistics
        avg_score = top_matches['ONET_Match_Score'].mean()
        
        recommendations[participant] = {
            'all_matches': top_matches,
            'excellent_matches': excellent_matches,
            'good_matches': good_matches, 
            'fair_matches': fair_matches,
            'avg_score': avg_score,
            'top_domains': top_domains,
            'holland_code': top_matches['Holland_Code'].iloc[0] if len(top_matches) > 0 else 'Unknown',
            'primary_interest': top_matches['Primary_Interest'].iloc[0] if len(top_matches) > 0 else 'Unknown',
            'interest_pattern': top_matches['Interest_Pattern'].iloc[0] if len(top_matches) > 0 else 'Unknown',
            'total_recommendations': len(top_matches)
        }
    
    return recommendations

def create_onet_visualizations(participant_onet, career_data, recommendations):
    """Create O*NET Interest Profiler visualizations"""
    print("📊 Creating O*NET Interest Profiler visualizations...")
    
    os.makedirs('onet_analysis_output', exist_ok=True)
    
    # 1. Interest Profile Radar Charts
    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    n_participants = len(participant_onet)
    n_cols = 3
    n_rows = (n_participants + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), subplot_kw=dict(projection='polar'))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for i, (_, participant) in enumerate(participant_onet.iterrows()):
        if i < len(axes):
            ax = axes[i]
            
            # Radar chart data
            scores = [participant[col] for col in riasec_cols]
            angles = np.linspace(0, 2 * np.pi, len(riasec_cols), endpoint=False)
            scores.append(scores[0])  # Close the polygon
            angles = np.append(angles, angles[0])
            
            ax.plot(angles, scores, 'o-', linewidth=2, color=colors[i % len(colors)])
            ax.fill(angles, scores, alpha=0.25, color=colors[i % len(colors)])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([col[0] for col in riasec_cols])
            ax.set_ylim(0, 100)
            ax.set_title(f'{participant["Participant"]}\n{participant["Holland_Code"]} - {participant["Interest_Pattern"]}', 
                        fontsize=10, fontweight='bold', pad=20)
            ax.grid(True)
    
    # Remove empty subplots
    for i in range(len(participant_onet), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('O*NET Interest Profiler - Individual Profiles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('onet_analysis_output/onet_interest_profiles.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Interest Level Distribution
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Interest level counts
    interest_levels = []
    for col in riasec_cols:
        interest_levels.extend(participant_onet[f'{col}_Level'].tolist())
    
    level_counts = Counter(interest_levels)
    ax1.bar(level_counts.keys(), level_counts.values(), color='skyblue')
    ax1.set_title('Interest Level Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    
    # Primary interest distribution
    primary_dist = participant_onet['Primary_Interest'].value_counts()
    ax2.pie(primary_dist.values, labels=primary_dist.index, autopct='%1.1f%%', colors=colors[:len(primary_dist)])
    ax2.set_title('Primary Interest Area Distribution', fontweight='bold')
    
    # Average interest scores
    avg_scores = participant_onet[riasec_cols].mean()
    ax3.bar(range(len(riasec_cols)), avg_scores.values, color=colors)
    ax3.set_xticks(range(len(riasec_cols)))
    ax3.set_xticklabels([col[0] for col in riasec_cols])
    ax3.set_title('Average Interest Scores (0-100)', fontweight='bold')
    ax3.set_ylabel('Score')
    
    # Match score distribution
    all_scores = []
    for participant, data in recommendations.items():
        all_scores.extend(data['all_matches']['ONET_Match_Score'].tolist())
    
    ax4.hist(all_scores, bins=15, color='lightcoral', alpha=0.7)
    ax4.set_title('O*NET Match Score Distribution', fontweight='bold')
    ax4.set_xlabel('Match Score')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('onet_analysis_output/onet_analysis_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ O*NET visualizations created!")

def generate_onet_reports(participant_onet, career_data, recommendations, matches_df):
    """Generate comprehensive O*NET Interest Profiler reports"""
    print("📝 Generating O*NET Interest Profiler reports...")
    
    # Summary CSV
    summary_data = []
    detailed_data = []
    
    for participant, rec_data in recommendations.items():
        all_matches = rec_data['all_matches']
        
        # Summary row
        summary_row = {
            'Participant': participant,
            'Holland_Code': rec_data['holland_code'],
            'Primary_Interest': rec_data['primary_interest'],
            'Interest_Pattern': rec_data['interest_pattern'],
            'Avg_Match_Score': rec_data['avg_score'],
            'Total_Recommendations': rec_data['total_recommendations'],
            'Excellent_Matches': len(rec_data['excellent_matches']),
            'Good_Matches': len(rec_data['good_matches']),
            'Fair_Matches': len(rec_data['fair_matches'])
        }
        
        # Add top 10 careers
        for i, (_, career) in enumerate(all_matches.head(10).iterrows()):
            summary_row[f'Top_Career_{i+1}'] = career['Career']
            summary_row[f'Domain_{i+1}'] = career['Domain']
            summary_row[f'Score_{i+1}'] = career['ONET_Match_Score']
        
        summary_data.append(summary_row)
        
        # Detailed rows
        for i, (_, career) in enumerate(all_matches.iterrows()):
            detailed_row = {
                'Participant': participant,
                'Rank': i + 1,
                'Career': career['Career'],
                'Domain': career['Domain'],
                'ONET_Match_Score': career['ONET_Match_Score'],
                'Correlation_Score': career['Correlation_Score'],
                'Cosine_Similarity': career['Cosine_Similarity'],
                'Euclidean_Distance': career['Euclidean_Distance'],
                'Holland_Code': career['Holland_Code'],
                'Match_Category': 'Excellent' if career['ONET_Match_Score'] >= 80 else 
                               'Good' if career['ONET_Match_Score'] >= 65 else
                               'Fair' if career['ONET_Match_Score'] >= 50 else 'Poor'
            }
            detailed_data.append(detailed_row)
    
    # Save CSV files
    pd.DataFrame(summary_data).to_csv('onet_analysis_output/onet_career_recommendations_summary.csv', index=False)
    pd.DataFrame(detailed_data).to_csv('onet_analysis_output/onet_career_recommendations_detailed.csv', index=False)
    
    # Text report
    report = []
    report.append("=" * 80)
    report.append("O*NET INTEREST PROFILER CAREER ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    report.append("")
    
    report.append("METHODOLOGY")
    report.append("-" * 50)
    report.append("• Based on O*NET Interest Profiler official methodology")
    report.append("• Uses correlation coefficient for career matching")
    report.append("• Interest levels: High (23-36), Moderate (15-22), Low (12-14), Very Low (0-11)")
    report.append("• Match categories: Excellent (80+), Good (65-79), Fair (50-64), Poor (<50)")
    report.append("")
    
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 50)
    report.append(f"• Participants Analyzed: {len(participant_onet)}")
    report.append(f"• Careers in Database: {len(career_data)}")
    report.append(f"• Career Domains: {career_data['Domain'].nunique()}")
    report.append(f"• Total Matches Calculated: {len(matches_df)}")
    report.append("")
    
    # Individual recommendations
    report.append("INDIVIDUAL O*NET INTEREST PROFILES")
    report.append("-" * 50)
    
    for participant, rec_data in recommendations.items():
        report.append(f"\n🎯 {participant.upper()}")
        report.append("-" * 30)
        
        participant_data = participant_onet[participant_onet['Participant'] == participant].iloc[0]
        riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
        
        report.append(f"Holland Code: {rec_data['holland_code']}")
        report.append(f"Primary Interest: {rec_data['primary_interest']}")
        report.append(f"Interest Pattern: {rec_data['interest_pattern']}")
        report.append(f"Interest Scores: " + " | ".join([f"{col[0]}:{participant_data[col]:.1f}" for col in riasec_cols]))
        
        # Interest levels
        high_areas = participant_data['High_Interest_Areas']
        moderate_areas = participant_data['Moderate_Interest_Areas']
        report.append(f"High Interest Areas: {high_areas}")
        report.append(f"Moderate Interest Areas: {moderate_areas}")
        
        report.append(f"Average Match Score: {rec_data['avg_score']:.1f}")
        report.append("")
        
        # Match categories
        excellent = rec_data['excellent_matches']
        good = rec_data['good_matches']
        fair = rec_data['fair_matches']
        
        if len(excellent) > 0:
            report.append("🌟 EXCELLENT MATCHES (80+ Score):")
            for i, (_, career) in enumerate(excellent.head(5).iterrows()):
                report.append(f"    {i+1}. {career['Career']} ({career['Domain']}) - {career['ONET_Match_Score']:.1f}")
            report.append("")
        
        if len(good) > 0:
            report.append("✅ GOOD MATCHES (65-79 Score):")
            for i, (_, career) in enumerate(good.head(5).iterrows()):
                report.append(f"    {i+1}. {career['Career']} ({career['Domain']}) - {career['ONET_Match_Score']:.1f}")
            report.append("")
        
        if len(fair) > 0:
            report.append("🔶 FAIR MATCHES (50-64 Score):")
            for i, (_, career) in enumerate(fair.head(3).iterrows()):
                report.append(f"    {i+1}. {career['Career']} ({career['Domain']}) - {career['ONET_Match_Score']:.1f}")
            report.append("")
    
    # Save text report
    with open('onet_analysis_output/ONET_Interest_Profiler_Report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print("✓ O*NET reports generated!")

def main():
    """Main O*NET Interest Profiler analysis function"""
    print("🚀 O*NET INTEREST PROFILER CAREER ANALYSIS")
    print("=" * 60)
    
    try:
        # Load data
        riasec_responses, careers_df = load_and_prepare_data()
        
        # Convert responses using O*NET methodology
        participant_onet = convert_onet_interest_scores(riasec_responses)
        
        # Calculate matches using O*NET approach
        matches_df = calculate_onet_career_matches(participant_onet, careers_df)
        
        # Generate O*NET recommendations
        recommendations = generate_onet_recommendations(matches_df)
        
        # Create visualizations
        create_onet_visualizations(participant_onet, careers_df, recommendations)
        
        # Generate reports
        generate_onet_reports(participant_onet, careers_df, recommendations, matches_df)
        
        # Save data files
        participant_onet.to_csv('onet_analysis_output/onet_participant_profiles.csv', index=False)
        matches_df.to_csv('onet_analysis_output/onet_all_career_matches.csv', index=False)
        
        print("\n✅ O*NET INTEREST PROFILER ANALYSIS COMPLETED!")
        print("=" * 60)
        print("📁 Generated Files:")
        print("  📊 Visualizations:")
        print("    • onet_interest_profiles.png")
        print("    • onet_analysis_overview.png")
        print("  📄 Reports:")
        print("    • onet_career_recommendations_summary.csv")
        print("    • onet_career_recommendations_detailed.csv")
        print("    • ONET_Interest_Profiler_Report.txt")
        print("  📋 Data Files:")
        print("    • onet_participant_profiles.csv")
        print("    • onet_all_career_matches.csv")
        
        # Summary statistics
        avg_match_score = np.mean([r['avg_score'] for r in recommendations.values()])
        total_excellent = sum(len(r['excellent_matches']) for r in recommendations.values())
        total_good = sum(len(r['good_matches']) for r in recommendations.values())
        
        print(f"\n📈 O*NET SUMMARY STATISTICS:")
        print(f"  • Participants: {len(participant_onet)}")
        print(f"  • Average Match Score: {avg_match_score:.1f}/100")
        print(f"  • Total Excellent Matches: {total_excellent}")
        print(f"  • Total Good Matches: {total_good}")
        
        # Sample result
        sample_participant = list(recommendations.keys())[0]
        sample_career = recommendations[sample_participant]['all_matches'].iloc[0]
        print(f"\n🎯 Sample O*NET Result:")
        print(f"  {sample_participant} → {sample_career['Career']}")
        print(f"  O*NET Match Score: {sample_career['ONET_Match_Score']:.1f}/100")
        
        return recommendations, participant_onet, careers_df
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()