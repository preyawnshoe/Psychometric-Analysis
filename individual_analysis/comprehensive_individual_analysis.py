"""
Comprehensive Individual Analysis
Combines RIASEC, Goal Orientation, Self-Concept data for complete individual profiles
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_all_datasets():
    """Load all analysis datasets"""
    print("Loading all available datasets...")
    
    datasets = {}
    base_path = r'C:\Users\p2var\vandana'
    
    # RIASEC-Goal Orientation data
    riasec_goal_path = os.path.join(base_path, 'goal_riasec_analysis', 'combined_riasec_goal_data_final.csv')
    if os.path.exists(riasec_goal_path):
        datasets['riasec_goal'] = pd.read_csv(riasec_goal_path)
        print(f"✓ Loaded RIASEC-Goal data: {len(datasets['riasec_goal'])} participants")
    
    # RIASEC-Self Concept data
    riasec_sc_path = os.path.join(base_path, 'riasec_self_concept_analysis', 'combined_riasec_self_concept_data.csv')
    if os.path.exists(riasec_sc_path):
        datasets['riasec_self_concept'] = pd.read_csv(riasec_sc_path)
        print(f"✓ Loaded RIASEC-Self Concept data: {len(datasets['riasec_self_concept'])} participants")
    
    # Goal-Self Concept data
    goal_sc_path = os.path.join(base_path, 'goal_self_concept_analysis', 'combined_goal_self_concept_data.csv')
    if os.path.exists(goal_sc_path):
        datasets['goal_self_concept'] = pd.read_csv(goal_sc_path)
        print(f"✓ Loaded Goal-Self Concept data: {len(datasets['goal_self_concept'])} participants")
    
    # Individual statistics
    stats_path = os.path.join(base_path, 'goal_riasec_analysis', 'individual_participant_statistics.csv')
    if os.path.exists(stats_path):
        datasets['individual_stats'] = pd.read_csv(stats_path)
        print(f"✓ Loaded Individual Statistics: {len(datasets['individual_stats'])} participants")
    
    # Individual profiles from different analyses
    profiles_path1 = os.path.join(base_path, 'riasec_self_concept_analysis', 'individual_profiles.csv')
    if os.path.exists(profiles_path1):
        datasets['riasec_sc_profiles'] = pd.read_csv(profiles_path1)
        print(f"✓ Loaded RIASEC-SC Profiles: {len(datasets['riasec_sc_profiles'])} participants")
    
    profiles_path2 = os.path.join(base_path, 'goal_self_concept_analysis', 'individual_profiles.csv')
    if os.path.exists(profiles_path2):
        datasets['goal_sc_profiles'] = pd.read_csv(profiles_path2)
        print(f"✓ Loaded Goal-SC Profiles: {len(datasets['goal_sc_profiles'])} participants")
    
    return datasets

def standardize_participant_names(datasets):
    """Standardize participant names across all datasets"""
    print("Standardizing participant names...")
    
    def clean_name(name):
        if pd.isna(name):
            return name
        return str(name).lower().strip()
    
    # Apply name cleaning to all datasets
    for key, df in datasets.items():
        if 'participant' in df.columns:
            df['participant'] = df['participant'].apply(clean_name)
            print(f"✓ Cleaned names in {key}")
    
    return datasets

def create_master_participant_list(datasets):
    """Create master list of all unique participants"""
    all_participants = set()
    
    for key, df in datasets.items():
        if 'participant' in df.columns:
            participants = set(df['participant'].dropna().unique())
            all_participants.update(participants)
            print(f"✓ {key}: {len(participants)} unique participants")
    
    master_list = sorted(list(all_participants))
    print(f"\n📋 MASTER PARTICIPANT LIST: {len(master_list)} total unique participants")
    for i, name in enumerate(master_list, 1):
        print(f"{i:2d}. {name.title()}")
    
    return master_list

def generate_comprehensive_profiles(datasets, master_list):
    """Generate comprehensive individual profiles"""
    print("\n🔍 GENERATING COMPREHENSIVE INDIVIDUAL PROFILES")
    print("=" * 60)
    
    comprehensive_profiles = []
    
    for participant in master_list:
        profile = {
            'participant': participant,
            'participant_title': participant.title()
        }
        
        # RIASEC Data
        if 'riasec_goal' in datasets:
            riasec_data = datasets['riasec_goal'][datasets['riasec_goal']['participant'] == participant]
            if not riasec_data.empty:
                row = riasec_data.iloc[0]
                profile.update({
                    'riasec_realistic': row.get('Realistic', np.nan),
                    'riasec_investigative': row.get('Investigative', np.nan),
                    'riasec_artistic': row.get('Artistic', np.nan),
                    'riasec_social': row.get('Social', np.nan),
                    'riasec_enterprising': row.get('Enterprising', np.nan),
                    'riasec_conventional': row.get('Conventional', np.nan),
                    'riasec_total': row.get('Total_RIASEC', np.nan),
                    'riasec_dominant': row.get('Dominant_RIASEC', 'Unknown'),
                    'cluster_assignment': row.get('Cluster', np.nan)
                })
                
                # Calculate RIASEC Holland Code
                riasec_scores = {
                    'R': row.get('Realistic', 0),
                    'I': row.get('Investigative', 0),
                    'A': row.get('Artistic', 0),
                    'S': row.get('Social', 0),
                    'E': row.get('Enterprising', 0),
                    'C': row.get('Conventional', 0)
                }
                sorted_riasec = sorted(riasec_scores.items(), key=lambda x: x[1], reverse=True)
                profile['holland_code'] = ''.join([item[0] for item in sorted_riasec[:3]])
        
        # Goal Orientation Data
        if 'riasec_goal' in datasets:
            goal_data = datasets['riasec_goal'][datasets['riasec_goal']['participant'] == participant]
            if not goal_data.empty:
                row = goal_data.iloc[0]
                profile.update({
                    'goal_performance_approach': row.get('Performance_Approach', np.nan),
                    'goal_mastery_approach': row.get('Mastery_Approach', np.nan),
                    'goal_performance_avoidance': row.get('Performance_Avoidance', np.nan),
                    'goal_mastery_avoidance': row.get('Mastery_Avoidance', np.nan),
                    'goal_total': row.get('Total_Goal_Orientation', np.nan),
                    'goal_dominant': row.get('Dominant_Goal', 'Unknown')
                })
                
                # Goal orientation classification
                ma = row.get('Mastery_Approach', 0)
                pa = row.get('Performance_Approach', 0)
                mv = row.get('Mastery_Avoidance', 0)
                pv = row.get('Performance_Avoidance', 0)
                
                if ma > max(pa, mv, pv):
                    goal_type = "Mastery-Focused Learner"
                elif pa > max(ma, mv, pv):
                    goal_type = "Performance-Driven Achiever"
                elif mv > max(ma, pa, pv):
                    goal_type = "Mastery-Anxious Perfectionist"
                else:
                    goal_type = "Performance-Anxious Avoider"
                
                profile['goal_orientation_type'] = goal_type
        
        # Self-Concept Data
        if 'riasec_self_concept' in datasets:
            sc_data = datasets['riasec_self_concept'][datasets['riasec_self_concept']['participant'] == participant]
            if not sc_data.empty:
                row = sc_data.iloc[0]
                profile.update({
                    'sc_health_sex': row.get('Health_Sex_Appropriateness', np.nan),
                    'sc_abilities': row.get('Abilities', np.nan),
                    'sc_confidence': row.get('Self_Confidence', np.nan),
                    'sc_acceptance': row.get('Self_Acceptance', np.nan),
                    'sc_worthiness': row.get('Worthiness', np.nan),
                    'sc_time_perspective': row.get('Present_Past_Future', np.nan),
                    'sc_beliefs': row.get('Beliefs_Convictions', np.nan),
                    'sc_shame_guilt': row.get('Shame_Guilt', np.nan),
                    'sc_sociability': row.get('Sociability', np.nan),
                    'sc_emotional': row.get('Emotional', np.nan),
                    'sc_total': row.get('Total_Self_Concept', np.nan)
                })
                
                # Self-concept classification
                total_sc = row.get('Total_Self_Concept', 0)
                if total_sc >= 3.5:
                    sc_type = "High Self-Concept (Strong)"
                elif total_sc >= 3.0:
                    sc_type = "Good Self-Concept (Positive)"
                elif total_sc >= 2.5:
                    sc_type = "Moderate Self-Concept (Balanced)"
                else:
                    sc_type = "Low Self-Concept (Needs Support)"
                
                profile['self_concept_type'] = sc_type
        
        # Individual Statistics
        if 'individual_stats' in datasets:
            stats_data = datasets['individual_stats'][datasets['individual_stats']['participant'] == participant]
            if not stats_data.empty:
                row = stats_data.iloc[0]
                profile.update({
                    'riasec_consistency': row.get('riasec_consistency', np.nan),
                    'goal_consistency': row.get('goal_consistency', np.nan),
                    'approach_vs_avoidance': row.get('approach_vs_avoidance', np.nan),
                    'performance_vs_mastery': row.get('performance_vs_mastery', np.nan)
                })
        
        # Profile Analysis
        if 'riasec_sc_profiles' in datasets:
            prof_data = datasets['riasec_sc_profiles'][datasets['riasec_sc_profiles']['participant'] == participant]
            if not prof_data.empty:
                row = prof_data.iloc[0]
                profile.update({
                    'riasec_code_alt': row.get('riasec_code', ''),
                    'sc_highest_domain': row.get('self_concept_highest', 'Unknown'),
                    'sc_lowest_domain': row.get('self_concept_lowest', 'Unknown')
                })
        
        comprehensive_profiles.append(profile)
    
    return pd.DataFrame(comprehensive_profiles)

def add_insights_and_recommendations(profiles_df):
    """Add insights and recommendations to profiles"""
    print("Adding psychological insights and recommendations...")
    
    insights = []
    recommendations = []
    
    for _, profile in profiles_df.iterrows():
        participant_insights = []
        participant_recommendations = []
        
        # RIASEC-based insights
        holland_code = profile.get('holland_code', '')
        if len(holland_code) >= 3:
            if holland_code[0] == 'R':
                participant_insights.append("Practical, hands-on learner")
                participant_recommendations.append("Engage in experiential learning and technical projects")
            elif holland_code[0] == 'I':
                participant_insights.append("Analytical, research-oriented thinker")
                participant_recommendations.append("Pursue independent research and problem-solving activities")
            elif holland_code[0] == 'A':
                participant_insights.append("Creative, expressive individual")
                participant_recommendations.append("Seek creative outlets and artistic expression opportunities")
            elif holland_code[0] == 'S':
                participant_insights.append("People-oriented, helping personality")
                participant_recommendations.append("Engage in mentoring, teaching, or counseling activities")
            elif holland_code[0] == 'E':
                participant_insights.append("Leadership-oriented, entrepreneurial")
                participant_recommendations.append("Take on leadership roles and business-oriented projects")
            elif holland_code[0] == 'C':
                participant_insights.append("Detail-oriented, systematic organizer")
                participant_recommendations.append("Excel in structured, organized, data-driven tasks")
        
        # Goal orientation insights
        goal_type = profile.get('goal_orientation_type', '')
        if goal_type and isinstance(goal_type, str):
            if 'Mastery-Focused' in goal_type:
                participant_insights.append("Deep learning orientation, intrinsically motivated")
                participant_recommendations.append("Set learning-focused goals, explore subjects deeply")
            elif 'Performance-Driven' in goal_type:
                participant_insights.append("Achievement-oriented, competitive motivation")
                participant_recommendations.append("Set challenging performance targets, competitive environments")
        
        # Self-concept insights
        sc_type = profile.get('self_concept_type', '')
        sc_total = profile.get('sc_total', 0)
        if sc_type and isinstance(sc_type, str):
            if 'High' in sc_type:
                participant_insights.append("Strong self-confidence and self-worth")
                participant_recommendations.append("Leverage confidence for leadership and challenging goals")
            elif 'Low' in sc_type:
                participant_insights.append("May benefit from confidence-building activities")
                participant_recommendations.append("Focus on strength-building and positive self-reflection")
        
        # Combination insights
        riasec_dom = profile.get('riasec_dominant', '')
        goal_dom = profile.get('goal_dominant', '')
        if riasec_dom == 'Investigative' and 'Mastery' in goal_dom:
            participant_insights.append("Natural researcher with strong learning drive")
            participant_recommendations.append("Pursue advanced academic or research opportunities")
        elif riasec_dom == 'Enterprising' and 'Performance' in goal_dom:
            participant_insights.append("Business-minded achiever with competitive drive")
            participant_recommendations.append("Consider entrepreneurial or leadership development programs")
        
        insights.append(" | ".join(participant_insights) if participant_insights else "Balanced profile")
        recommendations.append(" | ".join(participant_recommendations) if participant_recommendations else "Continue well-rounded development")
    
    profiles_df['key_insights'] = insights
    profiles_df['recommendations'] = recommendations
    
    return profiles_df

def create_summary_statistics(profiles_df):
    """Create summary statistics for the analysis"""
    print("\n📊 CREATING SUMMARY STATISTICS")
    print("=" * 50)
    
    stats = {}
    
    # RIASEC distribution
    riasec_dominant = profiles_df['riasec_dominant'].value_counts()
    print("\n🎯 RIASEC Type Distribution:")
    for riasec_type, count in riasec_dominant.items():
        percentage = (count / len(profiles_df)) * 100
        print(f"   {riasec_type}: {count} participants ({percentage:.1f}%)")
    
    # Goal orientation distribution
    goal_dominant = profiles_df['goal_dominant'].value_counts()
    print("\n🎯 Goal Orientation Distribution:")
    for goal_type, count in goal_dominant.items():
        percentage = (count / len(profiles_df)) * 100
        print(f"   {goal_type.replace('_', ' ')}: {count} participants ({percentage:.1f}%)")
    
    # Self-concept levels
    sc_types = profiles_df['self_concept_type'].value_counts()
    print("\n🎯 Self-Concept Level Distribution:")
    for sc_type, count in sc_types.items():
        percentage = (count / len(sc_types)) * 100
        print(f"   {sc_type}: {count} participants ({percentage:.1f}%)")
    
    # Average scores
    numeric_cols = ['riasec_total', 'goal_total', 'sc_total']
    print("\n📈 Average Scores:")
    for col in numeric_cols:
        if col in profiles_df.columns:
            mean_score = profiles_df[col].mean()
            std_score = profiles_df[col].std()
            print(f"   {col.replace('_', ' ').title()}: {mean_score:.3f} ± {std_score:.3f}")
    
    return stats

def save_comprehensive_analysis(profiles_df):
    """Save the comprehensive analysis"""
    print("\n💾 SAVING COMPREHENSIVE ANALYSIS")
    print("=" * 40)
    
    output_path = r'C:\Users\p2var\vandana\individual_analysis\comprehensive_individual_profiles.csv'
    profiles_df.to_csv(output_path, index=False)
    print(f"✓ Saved comprehensive profiles: {output_path}")
    
    # Create a readable summary table
    summary_columns = [
        'participant_title', 'holland_code', 'riasec_dominant', 'goal_orientation_type',
        'self_concept_type', 'sc_total', 'riasec_total', 'goal_total',
        'key_insights', 'recommendations'
    ]
    
    available_columns = [col for col in summary_columns if col in profiles_df.columns]
    summary_df = profiles_df[available_columns].copy()
    
    # Format numeric columns
    if 'sc_total' in summary_df.columns:
        summary_df['sc_total'] = summary_df['sc_total'].round(3)
    if 'riasec_total' in summary_df.columns:
        summary_df['riasec_total'] = summary_df['riasec_total'].round(3)
    if 'goal_total' in summary_df.columns:
        summary_df['goal_total'] = summary_df['goal_total'].round(3)
    
    summary_path = r'C:\Users\p2var\vandana\individual_analysis\individual_analysis_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary table: {summary_path}")
    
    # Create markdown report
    create_markdown_report(profiles_df, summary_df)
    
    return output_path, summary_path

def create_markdown_report(profiles_df, summary_df):
    """Create a markdown report"""
    report_path = r'C:\Users\p2var\vandana\individual_analysis\Individual_Analysis_Report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Comprehensive Individual Analysis Report\n\n")
        f.write(f"**Analysis Date:** {datetime.now().strftime('%B %d, %Y')}\n")
        f.write(f"**Total Participants:** {len(profiles_df)}\n\n")
        
        f.write("## Overview\n\n")
        f.write("This report provides comprehensive psychological profiles combining RIASEC career interests, ")
        f.write("achievement goal orientations, and multidimensional self-concept assessments for each participant.\n\n")
        
        f.write("## Individual Profiles\n\n")
        
        for _, profile in profiles_df.iterrows():
            name = profile.get('participant_title', 'Unknown')
            f.write(f"### {name}\n\n")
            
            # Basic profile
            f.write("**Profile Summary:**\n")
            f.write(f"- **Holland Code:** {profile.get('holland_code', 'N/A')}\n")
            f.write(f"- **Dominant RIASEC:** {profile.get('riasec_dominant', 'Unknown')}\n")
            f.write(f"- **Goal Orientation:** {profile.get('goal_orientation_type', 'Unknown')}\n")
            f.write(f"- **Self-Concept Level:** {profile.get('self_concept_type', 'Unknown')}\n\n")
            
            # Scores
            f.write("**Assessment Scores:**\n")
            if not pd.isna(profile.get('sc_total')):
                f.write(f"- **Self-Concept Total:** {profile.get('sc_total', 0):.3f}\n")
            if not pd.isna(profile.get('riasec_total')):
                f.write(f"- **RIASEC Total:** {profile.get('riasec_total', 0):.3f}\n")
            if not pd.isna(profile.get('goal_total')):
                f.write(f"- **Goal Orientation Total:** {profile.get('goal_total', 0):.3f}\n\n")
            
            # Insights and recommendations
            f.write("**Key Insights:**\n")
            f.write(f"{profile.get('key_insights', 'No specific insights available')}\n\n")
            
            f.write("**Recommendations:**\n")
            f.write(f"{profile.get('recommendations', 'Continue balanced development')}\n\n")
            
            f.write("---\n\n")
        
        f.write("## Data Sources\n\n")
        f.write("- RIASEC Career Interest Inventory\n")
        f.write("- Achievement Goal Orientation Questionnaire\n")
        f.write("- Multidimensional Self-Concept Assessment\n")
        f.write("- Individual Statistical Analysis\n\n")
        
        f.write("*Report generated using comprehensive psychological assessment integration*\n")
    
    print(f"✓ Saved markdown report: {report_path}")

def main():
    """Main comprehensive analysis function"""
    print("🔍 COMPREHENSIVE INDIVIDUAL ANALYSIS")
    print("=" * 60)
    print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    print("=" * 60)
    
    # Load all datasets
    datasets = load_all_datasets()
    
    if not datasets:
        print("❌ No datasets found! Please ensure analysis files exist.")
        return
    
    # Standardize names
    datasets = standardize_participant_names(datasets)
    
    # Create master participant list
    master_list = create_master_participant_list(datasets)
    
    # Generate comprehensive profiles
    profiles_df = generate_comprehensive_profiles(datasets, master_list)
    
    # Add insights and recommendations
    profiles_df = add_insights_and_recommendations(profiles_df)
    
    # Create summary statistics
    create_summary_statistics(profiles_df)
    
    # Save analysis
    output_path, summary_path = save_comprehensive_analysis(profiles_df)
    
    print("\n✅ COMPREHENSIVE INDIVIDUAL ANALYSIS COMPLETED!")
    print("=" * 60)
    print("Generated Files:")
    print(f"📋 Comprehensive Profiles: {output_path}")
    print(f"📊 Summary Table: {summary_path}")
    print(f"📄 Markdown Report: Individual_Analysis_Report.md")
    print("\nAnalysis includes:")
    print("• RIASEC career interest profiles")
    print("• Achievement goal orientations")
    print("• Self-concept assessments")
    print("• Individual statistical analysis")
    print("• Psychological insights and recommendations")

if __name__ == "__main__":
    main()