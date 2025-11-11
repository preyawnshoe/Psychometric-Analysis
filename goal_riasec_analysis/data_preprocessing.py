import pandas as pd
import numpy as np
import re
import os

def clean_participant_name(name):
    """Standardize participant names for matching"""
    if pd.isna(name) or name == '':
        return None
    
    # Convert to lowercase and remove extra spaces
    name = str(name).lower().strip()
    
    # Remove extra whitespaces
    name = re.sub(r'\s+', ' ', name)
    
    # Handle common variations
    name_mappings = {
        'vandana': 'vandana',
        'sanjana reddy pamuru': 'sanjana reddy pamuru',
        'priyanshu kumar': 'priyanshu kumar',
        'sanskar singhal': 'sanskar singhal',
        'ambarish chatterjee': 'ambarish chatterjee',
        'navya dhiman': 'navya dhiman',
        'hardhik': 'hardhik',
        'ahana sadh': 'ahana sadh',
        'navya ennam': 'navya ennam',
        'brinda': 'brinda',
        'aakash guduru': 'aakash guduru',
        'debajyoti banerjee': 'debajyoti banerjee',
        'sabyasachi': 'sabyasachi',
        'rudresh joshi': 'rudresh joshi',
        'durga': 'durga',
        'oishika sarkar': 'oishika sarkar'
    }
    
    return name_mappings.get(name, name)

def process_riasec_data():
    """Process RIASEC survey data and calculate RIASEC scores"""
    print("Processing RIASEC data...")
    
    # Read RIASEC data
    riasec_df = pd.read_csv(r'C:\Users\p2var\vandana\RIASEC\responses.csv')
    
    # Clean column names
    riasec_df.columns = riasec_df.columns.str.strip()
    
    # Extract participant names
    participant_col = [col for col in riasec_df.columns if 'Name Of the Participant' in col][0]
    riasec_df['participant'] = riasec_df[participant_col].apply(clean_participant_name)
    
    # Remove rows without participant names
    riasec_df = riasec_df.dropna(subset=['participant'])
    
    # RIASEC question mapping (based on Holland's theory)
    # Questions 1-42 mapped to RIASEC dimensions
    riasec_mapping = {
        'Realistic': [1, 7, 14, 22, 30, 32, 37],  # Hands-on, practical
        'Investigative': [2, 11, 18, 21, 26, 33, 39],  # Scientific, analytical
        'Artistic': [3, 8, 17, 23, 27, 31, 41],  # Creative, expressive
        'Social': [4, 12, 13, 20, 28, 34, 40],  # Helping, teaching
        'Enterprising': [5, 10, 16, 19, 29, 36, 42],  # Leading, persuading
        'Conventional': [6, 9, 15, 24, 25, 35, 38]  # Organizing, detail-oriented
    }
    
    # Convert responses to numeric values
    response_mapping = {'Yes': 2, 'May be': 1, 'No': 0}
    
    # Get question columns (columns 3 onwards, excluding timestamp and name)
    question_cols = [col for col in riasec_df.columns if col.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) or 
                     any(str(i) + '.' in col for i in range(10, 43))]
    
    # Convert responses to numeric
    for col in question_cols:
        riasec_df[col] = riasec_df[col].map(response_mapping)
    
    # Calculate RIASEC scores
    riasec_scores = {}
    
    for dimension, questions in riasec_mapping.items():
        score_cols = []
        for q_num in questions:
            # Find the column that corresponds to this question number
            for col in question_cols:
                if col.startswith(f'{q_num}.'):
                    score_cols.append(col)
                    break
        
        if score_cols:
            riasec_df[dimension] = riasec_df[score_cols].mean(axis=1)
        else:
            riasec_df[dimension] = 0
    
    # Calculate total RIASEC score
    riasec_dimensions = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    riasec_df['Total_RIASEC'] = riasec_df[riasec_dimensions].mean(axis=1)
    
    # Select relevant columns
    riasec_processed = riasec_df[['participant'] + riasec_dimensions + ['Total_RIASEC']].copy()
    
    print(f"RIASEC data processed: {len(riasec_processed)} participants")
    return riasec_processed

def process_goal_orientation_data():
    """Process goal orientation survey data"""
    print("Processing Goal Orientation data...")
    
    # Read goal orientation data
    goal_df = pd.read_csv(r'C:\Users\p2var\vandana\goal_orientation\responses.csv')
    
    # Clean participant names
    goal_df['participant'] = goal_df['Name Of the Participant'].apply(clean_participant_name)
    
    # Remove rows without participant names
    goal_df = goal_df.dropna(subset=['participant'])
    
    # Define response mapping
    response_mapping = {
        'Strongly Agree (+2)': 2, 'Strongly Agree': 2,
        'Agree (+1)': 1, 'Agree': 1,
        'No Opinion (0)': 0, 'No Opinion': 0,
        'Disagree (-1)': -1, 'Disagree': -1,
        'Strongly Disagree (-2)': -2, 'Strongly Disagree': -2
    }
    
    # Goal orientation dimensions (based on 2x2 achievement goal theory)
    goal_dimensions = {
        'Performance_Approach': [1, 2, 3, 4, 5, 6],  # Questions about outperforming others
        'Mastery_Approach': [7, 8, 9, 10, 11, 12],   # Questions about learning and mastery
        'Performance_Avoidance': [13, 14, 15, 16, 17, 18],  # Questions about avoiding poor performance
        'Mastery_Avoidance': [19, 20, 21]  # Questions about avoiding incomplete understanding
    }
    
    # Get question columns (exclude timestamp and participant)
    question_cols = [col for col in goal_df.columns if col not in ['Timestamp', 'Name Of the Participant', 'participant']]
    
    # Convert responses to numeric
    for col in question_cols:
        goal_df[col] = goal_df[col].map(response_mapping)
    
    # Calculate goal orientation scores
    for dimension, question_indices in goal_dimensions.items():
        relevant_cols = []
        for i, col in enumerate(question_cols):
            if i + 1 in question_indices:  # +1 because question_indices are 1-based
                relevant_cols.append(col)
        
        if relevant_cols:
            # Convert negative scores to positive scale (add 2 to shift from -2,2 to 0,4)
            goal_df[dimension] = (goal_df[relevant_cols] + 2).mean(axis=1)
        else:
            goal_df[dimension] = 0
    
    # Calculate total goal orientation score
    goal_dim_cols = list(goal_dimensions.keys())
    goal_df['Total_Goal_Orientation'] = goal_df[goal_dim_cols].mean(axis=1)
    
    # Select relevant columns
    goal_processed = goal_df[['participant'] + goal_dim_cols + ['Total_Goal_Orientation']].copy()
    
    print(f"Goal Orientation data processed: {len(goal_processed)} participants")
    return goal_processed

def merge_datasets():
    """Merge RIASEC and goal orientation datasets"""
    print("Merging datasets...")
    
    # Process both datasets
    riasec_data = process_riasec_data()
    goal_data = process_goal_orientation_data()
    
    # Merge on participant
    combined_data = pd.merge(riasec_data, goal_data, on='participant', how='inner')
    
    print(f"Combined data: {len(combined_data)} participants with both RIASEC and Goal Orientation data")
    print(f"Participants: {combined_data['participant'].tolist()}")
    
    # Save combined dataset
    output_path = r'C:\Users\p2var\vandana\goal_riasec_analysis\combined_riasec_goal_data.csv'
    combined_data.to_csv(output_path, index=False)
    print(f"Combined dataset saved to: {output_path}")
    
    # Display basic statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Total participants: {len(combined_data)}")
    print(f"RIASEC dimensions: Realistic, Investigative, Artistic, Social, Enterprising, Conventional")
    print(f"Goal Orientation dimensions: Performance_Approach, Mastery_Approach, Performance_Avoidance, Mastery_Avoidance")
    
    print("\n=== RIASEC SCORE SUMMARY ===")
    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    print(combined_data[riasec_cols].describe().round(3))
    
    print("\n=== GOAL ORIENTATION SCORE SUMMARY ===")
    goal_cols = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    print(combined_data[goal_cols].describe().round(3))
    
    return combined_data

if __name__ == "__main__":
    # Set up the analysis environment
    os.chdir(r'C:\Users\p2var\vandana\goal_riasec_analysis')
    
    # Process and merge data
    combined_data = merge_datasets()
    
    print("\nData preprocessing completed successfully!")
    print("Next steps: Run correlation analysis and visualization scripts.")