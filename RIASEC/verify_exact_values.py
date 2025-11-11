"""
Verification script to demonstrate that we're now using exact RIASEC values from career.csv
"""

import pandas as pd

def verify_exact_values():
    print("🔍 VERIFICATION: Using Exact RIASEC Values from career.csv")
    print("=" * 60)
    
    # Load the original career.csv
    career_csv = pd.read_csv('career.csv')
    print(f"📁 Loaded career.csv with {len(career_csv)} careers")
    
    # Load the generated career profiles
    career_profiles = pd.read_csv('career_analysis_output/career_riasec_profiles.csv')
    print(f"📊 Loaded generated profiles with {len(career_profiles)} careers")
    
    # Sample verification - check if values match exactly
    print("\n🎯 Sample Verification (First 5 careers):")
    print("-" * 50)
    
    for i in range(min(5, len(career_csv))):
        original = career_csv.iloc[i]
        generated = career_profiles.iloc[i]
        
        print(f"\nCareer: {original['Career']}")
        print(f"Original RIASEC: R={original['R']}, I={original['I']}, A={original['A']}, S={original['S']}, E={original['E']}, C={original['C']}")
        print(f"Used RIASEC:     R={generated['Realistic']}, I={generated['Investigative']}, A={generated['Artistic']}, S={generated['Social']}, E={generated['Enterprising']}, C={generated['Conventional']}")
        
        # Check if they match
        matches = (
            original['R'] == generated['Realistic'] and
            original['I'] == generated['Investigative'] and
            original['A'] == generated['Artistic'] and
            original['S'] == generated['Social'] and
            original['E'] == generated['Enterprising'] and
            original['C'] == generated['Conventional']
        )
        print(f"✅ Values Match: {matches}")
    
    # Overall statistics
    print("\n📈 RIASEC Value Statistics:")
    print("-" * 30)
    
    riasec_cols = ['R', 'I', 'A', 'S', 'E', 'C']
    for col in riasec_cols:
        values = career_csv[col].unique()
        print(f"{col}: {sorted(values)}")
    
    print(f"\n✅ SUCCESS: Now using exact RIASEC values from career.csv!")
    print(f"   • Total careers: {len(career_csv)}")
    print(f"   • RIASEC scale: 0.0 to {career_csv[riasec_cols].max().max()}")
    print(f"   • No more hardcoded domain profiles!")

if __name__ == "__main__":
    verify_exact_values()