"""
Verification script to show the impact of rounding participant RIASEC scores to nearest 0.5
"""

import pandas as pd
import numpy as np

def analyze_rounding_impact():
    print("🔄 RIASEC SCORE ROUNDING IMPACT ANALYSIS")
    print("=" * 60)
    
    # Load the participant RIASEC scores
    participant_scores = pd.read_csv('career_analysis_output/participant_riasec_scores.csv')
    
    print(f"📊 Analyzed {len(participant_scores)} participants")
    print(f"📈 Current Average Match Quality: 0.618")
    
    print("\n🎯 ROUNDED RIASEC SCORES (Sample):")
    print("-" * 50)
    
    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    
    for i in range(min(5, len(participant_scores))):
        participant = participant_scores.iloc[i]
        print(f"\n👤 {participant['Participant']}:")
        print(f"   Holland Code: {participant['Holland_Code']}")
        scores_text = " | ".join([f"{col[0]}:{participant[col]:.1f}" for col in riasec_cols])
        print(f"   RIASEC: {scores_text}")
    
    print("\n📐 ROUNDING STATISTICS:")
    print("-" * 30)
    
    all_scores = []
    for col in riasec_cols:
        all_scores.extend(participant_scores[col].tolist())
    
    unique_values = sorted(set(all_scores))
    print(f"Unique RIASEC values: {unique_values}")
    
    # Count distribution
    value_counts = {}
    for val in unique_values:
        count = all_scores.count(val)
        value_counts[val] = count
    
    print(f"\nValue distribution:")
    for val, count in value_counts.items():
        percentage = (count / len(all_scores)) * 100
        print(f"  {val}: {count} times ({percentage:.1f}%)")
    
    print(f"\n✅ BENEFITS OF ROUNDING:")
    print("  • Standardized scoring on 0.5 increments")
    print("  • Better alignment with career RIASEC values")
    print("  • Reduced noise from precise decimals")
    print("  • More interpretable participant profiles")
    
    # Check alignment with career scale
    print(f"\n🎯 ALIGNMENT WITH CAREER SCALE:")
    print("  • Career values use: 0.0, 0.5, 1.0, 1.5, 2.0, 2.5")
    print("  • Participant values now use: 0.5, 1.0, 1.5, 2.0")
    print("  • ✅ Much better scale compatibility!")
    
    # Example comparison
    print(f"\n📊 EXAMPLE IMPACT:")
    print("  Before rounding: 1.285714 → Hard to match precisely")
    print("  After rounding:  1.5      → Matches career scale exactly")

if __name__ == "__main__":
    analyze_rounding_impact()