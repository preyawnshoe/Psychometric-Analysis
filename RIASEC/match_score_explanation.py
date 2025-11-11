"""
RIASEC Career Match Score Calculation Methodology
Detailed explanation of the multi-algorithm approach used in career matching
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

def explain_match_score_calculation():
    """
    Comprehensive explanation of how career match scores are calculated
    """
    
    print("🧮 RIASEC CAREER MATCH SCORE CALCULATION")
    print("=" * 60)
    
    print("\n📊 OVERVIEW:")
    print("The match score is calculated using a weighted combination of 4 different")
    print("similarity algorithms to ensure robust and reliable career recommendations.")
    print()
    
    print("🎯 ALGORITHM BREAKDOWN:")
    print("-" * 40)
    
    print("\n1️⃣ COSINE SIMILARITY (Weight: 30%)")
    print("   Purpose: Measures the angular similarity between RIASEC profiles")
    print("   Formula: cos(θ) = (A · B) / (|A| × |B|)")
    print("   Range: 0 to 1 (1 = perfect match)")
    print("   Best for: Overall profile shape matching")
    print()
    
    print("2️⃣ EUCLIDEAN SIMILARITY (Weight: 25%)")
    print("   Purpose: Measures direct distance between RIASEC scores")
    print("   Formula: similarity = 1 / (1 + √Σ(participant_i - career_i)²)")
    print("   Range: 0 to 1 (1 = identical scores)")
    print("   Best for: Exact score matching")
    print()
    
    print("3️⃣ PEARSON CORRELATION (Weight: 25%)")
    print("   Purpose: Measures linear relationship pattern")
    print("   Formula: r = Σ((x_i - x̄)(y_i - ȳ)) / √(Σ(x_i - x̄)² × Σ(y_i - ȳ)²)")
    print("   Range: 0 to 1 (negative correlations set to 0)")
    print("   Best for: Pattern similarity regardless of scale")
    print()
    
    print("4️⃣ WEIGHTED SCORE (Weight: 20%)")
    print("   Purpose: Emphasizes participant's strongest RIASEC areas")
    print("   Formula: Σ(participant_weight_i × career_score_i)")
    print("   Where: participant_weight_i = participant_score_i / Σ(participant_scores)")
    print("   Best for: Matching careers that align with personal strengths")
    print()
    
    print("🔢 FINAL COMBINED SCORE:")
    print("Combined Score = (Cosine × 0.30) + (Euclidean × 0.25) + (Correlation × 0.25) + (Weighted × 0.20)")
    print()
    
    return True

def demonstrate_calculation_example():
    """
    Demonstrate the calculation with a real example
    """
    
    print("📋 CALCULATION EXAMPLE:")
    print("=" * 50)
    
    # Example participant and career scores
    participant_scores = np.array([2.43, 2.71, 2.29, 2.43, 2.43, 2.14])  # Sanskar Singhal
    career_scores = np.array([0.5, 1.0, 3.0, 1.0, 1.5, 0.5])  # Designer & Master Cutter
    
    riasec_types = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    
    print("\n🧑‍💼 PARTICIPANT: Sanskar Singhal (Holland Code: ARS)")
    for i, riasec_type in enumerate(riasec_types):
        print(f"   {riasec_type}: {participant_scores[i]:.2f}")
    
    print("\n💼 CAREER: Designer & Master Cutter")
    for i, riasec_type in enumerate(riasec_types):
        print(f"   {riasec_type}: {career_scores[i]:.2f}")
    
    print("\n🔍 STEP-BY-STEP CALCULATION:")
    print("-" * 30)
    
    # Method 1: Cosine Similarity
    cosine_sim = cosine_similarity(participant_scores.reshape(1, -1), 
                                 career_scores.reshape(1, -1))[0][0]
    print(f"\n1️⃣ Cosine Similarity = {cosine_sim:.6f}")
    print(f"   Contribution to final score: {cosine_sim * 0.3:.6f} (×0.30)")
    
    # Method 2: Euclidean Distance
    euclidean_dist = np.linalg.norm(participant_scores - career_scores)
    euclidean_sim = 1 / (1 + euclidean_dist)
    print(f"\n2️⃣ Euclidean Distance = {euclidean_dist:.6f}")
    print(f"   Euclidean Similarity = 1/(1+{euclidean_dist:.6f}) = {euclidean_sim:.6f}")
    print(f"   Contribution to final score: {euclidean_sim * 0.25:.6f} (×0.25)")
    
    # Method 3: Correlation
    correlation = pearsonr(participant_scores, career_scores)[0]
    correlation = max(0, correlation)
    print(f"\n3️⃣ Pearson Correlation = {correlation:.6f}")
    print(f"   Contribution to final score: {correlation * 0.25:.6f} (×0.25)")
    
    # Method 4: Weighted Score
    participant_weights = participant_scores / np.sum(participant_scores)
    weighted_score = np.sum(participant_weights * career_scores)
    print(f"\n4️⃣ Weighted Score Calculation:")
    print(f"   Participant total score: {np.sum(participant_scores):.2f}")
    print(f"   Participant weights: {participant_weights}")
    print(f"   Weighted score = Σ(weights × career_scores) = {weighted_score:.6f}")
    print(f"   Contribution to final score: {weighted_score * 0.2:.6f} (×0.20)")
    
    # Final combined score
    combined_score = (cosine_sim * 0.3 + euclidean_sim * 0.25 + 
                     correlation * 0.25 + weighted_score * 0.2)
    
    print(f"\n🎯 FINAL COMBINED SCORE:")
    print(f"   {cosine_sim * 0.3:.6f} + {euclidean_sim * 0.25:.6f} + {correlation * 0.25:.6f} + {weighted_score * 0.2:.6f}")
    print(f"   = {combined_score:.6f}")
    print(f"   = {combined_score * 100:.2f}% match")
    
    return combined_score

def explain_interpretation():
    """
    Explain how to interpret match scores
    """
    
    print("\n📈 MATCH SCORE INTERPRETATION:")
    print("=" * 40)
    
    print("\n🎯 SCORE RANGES:")
    print("   90-100%: Excellent match - Highly recommended")
    print("   80-89%:  Good match - Recommended")
    print("   70-79%:  Moderate match - Consider exploring")
    print("   60-69%:  Fair match - May require additional consideration")
    print("   <60%:    Poor match - Not recommended")
    
    print("\n💡 WHY MULTIPLE ALGORITHMS?")
    print("   • Cosine Similarity: Handles different score scales well")
    print("   • Euclidean Distance: Rewards exact score matches")
    print("   • Correlation: Captures pattern similarity")
    print("   • Weighted Score: Emphasizes individual strengths")
    print("   • Combined approach reduces bias and increases reliability")
    
    print("\n🔍 WHAT MAKES A HIGH MATCH?")
    print("   ✅ Similar RIASEC profile shape")
    print("   ✅ Close individual dimension scores")
    print("   ✅ Strong alignment in participant's top areas")
    print("   ✅ Consistent patterns across all algorithms")
    
    print("\n⚠️  CONSIDERATIONS:")
    print("   • Scores above 85% indicate strong career-person fit")
    print("   • Multiple high-scoring careers suggest good options")
    print("   • Domain context should also be considered")
    print("   • Personal interests beyond RIASEC matter too")

def analyze_actual_results():
    """
    Analyze some actual results from the career analysis
    """
    
    print("\n📊 ACTUAL RESULTS ANALYSIS:")
    print("=" * 40)
    
    # Load actual results
    try:
        detailed_df = pd.read_csv('career_analysis_output/career_recommendations_detailed.csv')
        
        print(f"\n📈 MATCH SCORE STATISTICS:")
        print(f"   Highest Match: {detailed_df['Combined_Score'].max():.3f} ({detailed_df['Combined_Score'].max()*100:.1f}%)")
        print(f"   Lowest Match: {detailed_df['Combined_Score'].min():.3f} ({detailed_df['Combined_Score'].min()*100:.1f}%)")
        print(f"   Average Match: {detailed_df['Combined_Score'].mean():.3f} ({detailed_df['Combined_Score'].mean()*100:.1f}%)")
        print(f"   Standard Deviation: {detailed_df['Combined_Score'].std():.3f}")
        
        # Top matches
        top_matches = detailed_df.nlargest(5, 'Combined_Score')[['Participant', 'Career', 'Combined_Score']]
        print(f"\n🏆 TOP 5 MATCHES:")
        for _, match in top_matches.iterrows():
            print(f"   {match['Participant']} → {match['Career']}: {match['Combined_Score']:.3f} ({match['Combined_Score']*100:.1f}%)")
        
        # Distribution analysis
        score_ranges = {
            'Excellent (90-100%)': len(detailed_df[detailed_df['Combined_Score'] >= 0.9]),
            'Good (80-89%)': len(detailed_df[(detailed_df['Combined_Score'] >= 0.8) & (detailed_df['Combined_Score'] < 0.9)]),
            'Moderate (70-79%)': len(detailed_df[(detailed_df['Combined_Score'] >= 0.7) & (detailed_df['Combined_Score'] < 0.8)]),
            'Fair (60-69%)': len(detailed_df[(detailed_df['Combined_Score'] >= 0.6) & (detailed_df['Combined_Score'] < 0.7)]),
            'Poor (<60%)': len(detailed_df[detailed_df['Combined_Score'] < 0.6])
        }
        
        print(f"\n📊 SCORE DISTRIBUTION (Top 4 per participant):")
        for range_name, count in score_ranges.items():
            percentage = (count / len(detailed_df)) * 100
            print(f"   {range_name}: {count} recommendations ({percentage:.1f}%)")
        
    except FileNotFoundError:
        print("   (Analysis files not found - run the career analysis first)")

def main():
    """
    Main explanation function
    """
    
    # Core explanation
    explain_match_score_calculation()
    
    # Practical example
    demonstrate_calculation_example()
    
    # Interpretation guide
    explain_interpretation()
    
    # Real results analysis
    analyze_actual_results()
    
    print("\n" + "=" * 60)
    print("🎓 SUMMARY:")
    print("The match score combines 4 algorithms with specific weights to")
    print("provide robust, reliable career recommendations based on RIASEC")
    print("personality-career fit theory. Higher scores indicate better")
    print("alignment between individual interests and career requirements.")
    print("=" * 60)

if __name__ == "__main__":
    main()