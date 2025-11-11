import pandas as pd
import numpy as np

def convert_response_to_score(response):
    """Convert text responses to numerical scores"""
    if pd.isna(response) or response == '':
        return 0
    
    response = str(response).strip()
    
    # Handle responses with explicit scores in parentheses
    if '(+2)' in response or response == 'Strongly Agree':
        return 2
    elif '(+1)' in response or response == 'Agree':
        return 1
    elif '(0)' in response or response == 'No Opinion':
        return 0
    elif '(-1)' in response or response == 'Disagree':
        return -1
    elif '(-2)' in response or response == 'Strongly Disagree':
        return -2
    else:
        return 0

def analyze_goal_orientation(csv_file_path):
    """
    Analyze goal orientation survey responses and categorize participants
    """
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Define the categories and their corresponding question indices (0-based)
    categories = {
        'Performance_Approach': [1, 2, 3, 4, 5, 6],  # Questions about outperforming others
        'Mastery_Approach': [7, 8, 9, 10, 11, 12],   # Questions about learning and mastery
        'Performance_Avoidance': [13, 14, 15, 16, 17, 18],  # Questions about avoiding poor performance
        'Mastery_Avoidance': [19, 20, 21]            # Questions about worry about not learning enough
    }
    
    # Create a results dataframe
    results = []
    
    for index, row in df.iterrows():
        # Get participant name (last column)
        participant_name = row.iloc[-1] if pd.notna(row.iloc[-1]) and row.iloc[-1].strip() != '' else f"Participant_{index+1}"
        
        # Calculate scores for each category
        scores = {}
        for category, question_indices in categories.items():
            category_scores = []
            for q_idx in question_indices:
                if q_idx < len(row) - 1:  # Exclude timestamp and name columns
                    score = convert_response_to_score(row.iloc[q_idx])
                    category_scores.append(score)
            
            # Calculate average score for the category
            if category_scores:
                scores[category] = round(np.mean(category_scores), 2)
            else:
                scores[category] = 0
        
        # Sort orientations by score in descending order
        sorted_orientations = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create ranking strings
        orientation_ranking = " > ".join([f"{cat.replace('_', '-')}({score})" for cat, score in sorted_orientations])
        
        # Determine dominant orientation (highest score)
        max_score = max(scores.values())
        dominant_categories = [cat for cat, score in scores.items() if score == max_score]
        dominant_orientation = dominant_categories[0] if dominant_categories else "Undefined"
        
        # Add to results
        result_row = {
            'Participant_Name': participant_name,
            'Performance_Approach_Score': scores['Performance_Approach'],
            'Mastery_Approach_Score': scores['Mastery_Approach'],
            'Performance_Avoidance_Score': scores['Performance_Avoidance'],
            'Mastery_Avoidance_Score': scores['Mastery_Avoidance'],
            'Orientation_Ranking': orientation_ranking,
            'Dominant_Orientation': dominant_orientation,
            'Highest_Score': max_score,
            'Rank_1': sorted_orientations[0][0].replace('_', '-'),
            'Rank_1_Score': sorted_orientations[0][1],
            'Rank_2': sorted_orientations[1][0].replace('_', '-'),
            'Rank_2_Score': sorted_orientations[1][1],
            'Rank_3': sorted_orientations[2][0].replace('_', '-'),
            'Rank_3_Score': sorted_orientations[2][1],
            'Rank_4': sorted_orientations[3][0].replace('_', '-'),
            'Rank_4_Score': sorted_orientations[3][1]
        }
        results.append(result_row)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Add interpretation column
    def interpret_orientation(row):
        if row['Dominant_Orientation'] == 'Performance_Approach':
            return "Focuses on outperforming others and demonstrating superior ability"
        elif row['Dominant_Orientation'] == 'Mastery_Approach':
            return "Focuses on learning, understanding, and skill development"
        elif row['Dominant_Orientation'] == 'Performance_Avoidance':
            return "Focuses on avoiding appearing incompetent or performing poorly"
        elif row['Dominant_Orientation'] == 'Mastery_Avoidance':
            return "Worried about not learning enough or missing important knowledge"
        else:
            return "Mixed or unclear orientation"
    
    results_df['Interpretation'] = results_df.apply(interpret_orientation, axis=1)
    
    return results_df

def main():
    # Analyze the survey data
    results_df = analyze_goal_orientation('responses.csv')
    
    # Save to Excel file
    output_file = 'goal_orientation_results_ranked.xlsx'
    
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Main results sheet
        results_df.to_excel(writer, sheet_name='Goal_Orientation_Results', index=False)
        
        # Summary statistics sheet
        summary_stats = {
            'Category': ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance'],
            'Mean_Score': [
                results_df['Performance_Approach_Score'].mean(),
                results_df['Mastery_Approach_Score'].mean(),
                results_df['Performance_Avoidance_Score'].mean(),
                results_df['Mastery_Avoidance_Score'].mean()
            ],
            'Std_Deviation': [
                results_df['Performance_Approach_Score'].std(),
                results_df['Mastery_Approach_Score'].std(),
                results_df['Performance_Avoidance_Score'].std(),
                results_df['Mastery_Avoidance_Score'].std()
            ],
            'Count_as_Dominant': [
                (results_df['Dominant_Orientation'] == 'Performance_Approach').sum(),
                (results_df['Dominant_Orientation'] == 'Mastery_Approach').sum(),
                (results_df['Dominant_Orientation'] == 'Performance_Avoidance').sum(),
                (results_df['Dominant_Orientation'] == 'Mastery_Avoidance').sum()
            ]
        }
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df['Mean_Score'] = summary_df['Mean_Score'].round(2)
        summary_df['Std_Deviation'] = summary_df['Std_Deviation'].round(2)
        
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"Analysis complete! Results saved to '{output_file}'")
    print(f"\nTotal participants analyzed: {len(results_df)}")
    print("\nDominant orientations distribution:")
    print(results_df['Dominant_Orientation'].value_counts())
    print(f"\nPreview of ranking results:")
    preview_cols = ['Participant_Name', 'Orientation_Ranking', 'Highest_Score']
    print(results_df[preview_cols].head(10).to_string(index=False))

if __name__ == "__main__":
    main()