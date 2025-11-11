"""
Goal Orientation - Self Concept Cross-Analysis Data Preprocessing
Creates merged dataset for statistical analysis
"""

import pandas as pd
import numpy as np
import re

# Self-concept dimension mapping based on table.md
SELF_CONCEPT_DIMENSIONS = {
    'Health_Sex_Appropriateness': [
        {'question': 6, 'polarity': 'P'},
        {'question': 32, 'polarity': 'P'},
        {'question': 34, 'polarity': 'N'},
        {'question': 46, 'polarity': 'P'}
    ],
    'Abilities': [
        {'question': 4, 'polarity': 'P'},
        {'question': 8, 'polarity': 'P'},
        {'question': 12, 'polarity': 'N'},
        {'question': 22, 'polarity': 'N'},
        {'question': 34, 'polarity': 'P'},
        {'question': 36, 'polarity': 'N'},
        {'question': 37, 'polarity': 'N'},
        {'question': 40, 'polarity': 'P'}
    ],
    'Self_Confidence': [
        {'question': 7, 'polarity': 'P'},
        {'question': 9, 'polarity': 'P'},
        {'question': 14, 'polarity': 'N'},
        {'question': 16, 'polarity': 'N'},
        {'question': 42, 'polarity': 'P'}
    ],
    'Self_Acceptance': [
        {'question': 2, 'polarity': 'P'},
        {'question': 10, 'polarity': 'N'},
        {'question': 17, 'polarity': 'N'},
        {'question': 33, 'polarity': 'N'}
    ],
    'Worthiness': [
        {'question': 1, 'polarity': 'P'},
        {'question': 3, 'polarity': 'N'},
        {'question': 19, 'polarity': 'N'},
        {'question': 24, 'polarity': 'P'},
        {'question': 39, 'polarity': 'N'},
        {'question': 46, 'polarity': 'P'}
    ],
    'Present_Past_Future': [
        {'question': 18, 'polarity': 'P'},
        {'question': 21, 'polarity': 'P'},
        {'question': 25, 'polarity': 'P'},
        {'question': 26, 'polarity': 'P'},
        {'question': 38, 'polarity': 'P'}
    ],
    'Beliefs_Convictions': [
        {'question': 23, 'polarity': 'N'},
        {'question': 45, 'polarity': 'P'},
        {'question': 47, 'polarity': 'P'}
    ],
    'Shame_Guilt': [
        {'question': 5, 'polarity': 'N'},
        {'question': 13, 'polarity': 'N'},
        {'question': 27, 'polarity': 'N'},
        {'question': 28, 'polarity': 'N'},
        {'question': 48, 'polarity': 'N'}
    ],
    'Sociability': [
        {'question': 31, 'polarity': 'P'},
        {'question': 35, 'polarity': 'P'},
        {'question': 41, 'polarity': 'P'},
        {'question': 43, 'polarity': 'N'}
    ],
    'Emotional': [
        {'question': 11, 'polarity': 'N'},
        {'question': 15, 'polarity': 'N'},
        {'question': 30, 'polarity': 'N'},  # Based on updated mapping
        {'question': 49, 'polarity': 'N'}
    ]
}

def clean_participant_name(name):
    """Clean and standardize participant names"""
    if pd.isna(name):
        return None
    
    name = str(name).strip().lower()
    
    # Remove special characters and normalize spaces
    name = re.sub(r'[^\w\s]', '', name)
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

def convert_response_to_numeric(response):
    """Convert Likert scale responses to numeric values"""
    if pd.isna(response):
        return np.nan
    
    response_str = str(response).strip().lower()
    
    # Define mapping for different response formats
    response_mapping = {
        # Standard Likert scale
        'strongly agree': 5, 'agree': 4, 'undecided': 3, 'disagree': 2, 'strongly disagree': 1,
        'strongly_agree': 5, 'agree': 4, 'undecided': 3, 'disagree': 2, 'strongly_disagree': 1,
        'very true': 5, 'true': 4, 'neutral': 3, 'false': 2, 'very false': 1,
        
        # Goal orientation specific format (from the CSV)
        'strongly agree (+2)': 5, 'agree (+1)': 4, 'no opinion (0)': 3, 'disagree (-1)': 2, 'strongly disagree (-2)': 1,
        'strongly agree': 5, 'agree': 4, 'no opinion': 3, 'disagree': 2, 'strongly disagree': 1,
        
        # Numeric formats
        5: 5, 4: 4, 3: 3, 2: 2, 1: 1,
        '5': 5, '4': 4, '3': 3, '2': 2, '1': 1,
        '+2': 5, '+1': 4, '0': 3, '-1': 2, '-2': 1,
        2: 5, 1: 4, 0: 3, -1: 2, -2: 1  # For the numbered system
    }
    
    return response_mapping.get(response_str, response_mapping.get(response, np.nan))

def process_goal_orientation_data():
    """Process goal orientation survey data"""
    print("Processing Goal Orientation data...")
    
    # Load goal orientation responses
    goal_path = r'C:\Users\p2var\vandana\goal_orientation\responses.csv'
    goal_df = pd.read_csv(goal_path)
    
    print(f"Goal orientation raw data shape: {goal_df.shape}")
    print(f"Columns: {list(goal_df.columns)}")
    
    # Clean participant names
    goal_df['participant'] = goal_df['Name Of the Participant'].apply(clean_participant_name)
    goal_df = goal_df.dropna(subset=['participant'])
    
    # Goal orientation dimensions mapping based on Achievement Goal Theory
    # Questions are 1-indexed in the survey, 0-indexed in dataframe columns (excluding Timestamp)
    goal_categories = {
        'Performance_Approach': [1, 2, 3, 4, 5, 6],  # Questions about outperforming others
        'Mastery_Approach': [7, 8, 9, 10, 11, 12],   # Questions about learning and mastery
        'Performance_Avoidance': [13, 14, 15, 16, 17, 18],  # Questions about avoiding poor performance
        'Mastery_Avoidance': [19, 20, 21]            # Questions about worry about not learning enough
    }
    
    # Get question columns (excluding Timestamp and participant name)
    question_columns = [col for col in goal_df.columns if col not in ['Timestamp', 'Name Of the Participant', 'participant']]
    
    print(f"Found {len(question_columns)} question columns")
    
    participant_data = []
    
    for _, row in goal_df.iterrows():
        participant_scores = {'participant': row['participant']}
        
        # Calculate scores for each goal orientation dimension
        for dimension, question_indices in goal_categories.items():
            scores = []
            
            for q_idx in question_indices:
                if q_idx <= len(question_columns):
                    # Convert 1-based index to 0-based for column access
                    col_idx = q_idx - 1
                    q_col = question_columns[col_idx]
                    
                    # Get response and convert to numeric
                    response = convert_goal_response_to_numeric(row[q_col])
                    if not pd.isna(response):
                        scores.append(response)
            
            # Calculate mean score for this dimension
            if scores:
                participant_scores[dimension] = np.mean(scores)
            else:
                participant_scores[dimension] = np.nan
        
        participant_data.append(participant_scores)
    
    goal_processed = pd.DataFrame(participant_data)
    
    print(f"Goal Orientation data processed: {len(goal_processed)} participants")
    print(f"Dimensions: {[col for col in goal_processed.columns if col != 'participant']}")
    
    return goal_processed

def convert_goal_response_to_numeric(response):
    """Convert goal orientation responses to numeric values (Achievement Goal Theory scale)"""
    if pd.isna(response):
        return np.nan
    
    response_str = str(response).strip()
    
    # Goal orientation specific format (from the actual CSV data)
    response_mapping = {
        'Strongly Agree (+2)': 5, 'Agree (+1)': 4, 'No Opinion (0)': 3, 'Disagree (-1)': 2, 'Strongly Disagree (-2)': 1,
        'Strongly Agree': 5, 'Agree': 4, 'No Opinion': 3, 'Disagree': 2, 'Strongly Disagree': 1,
    }
    
    return response_mapping.get(response_str, np.nan)

def process_self_concept_data():
    """Process self-concept survey data based on the mapping in table.md"""
    print("Processing Self-Concept data...")
    
    # Read self-concept data
    self_concept_df = pd.read_csv(r'C:\Users\p2var\vandana\self_concept\responses.csv')
    
    print(f"Self-concept raw data shape: {self_concept_df.shape}")
    
    # Clean participant names
    self_concept_df['participant'] = self_concept_df['Name'].apply(clean_participant_name)
    self_concept_df = self_concept_df.dropna(subset=['participant'])
    
    # Get question columns (starting from column 8)
    question_columns = self_concept_df.columns[8:]  # Skip demographic columns
    
    participant_data = []
    
    for _, row in self_concept_df.iterrows():
        participant_scores = {'participant': row['participant']}
        
        # Calculate scores for each self-concept dimension
        for dimension, questions_info in SELF_CONCEPT_DIMENSIONS.items():
            scores = []
            
            for q_info in questions_info:
                question_num = q_info['question']
                polarity = q_info['polarity']
                
                # Find the corresponding column (questions are 1-indexed, columns are 0-indexed)
                if question_num <= len(question_columns):
                    col_idx = question_num - 1
                    q_col = question_columns[col_idx]
                    
                    # Get response and convert to numeric
                    response = convert_response_to_numeric(row[q_col])
                    
                    if not pd.isna(response):
                        # Apply polarity (reverse negative items)
                        if polarity == 'N':
                            response = 6 - response  # Reverse 5-point scale
                        scores.append(response)
            
            # Calculate mean score for this dimension
            if scores:
                participant_scores[dimension] = np.mean(scores)
            else:
                participant_scores[dimension] = np.nan
        
        participant_data.append(participant_scores)
    
    # Create DataFrame
    self_concept_data = pd.DataFrame(participant_data)
    
    # Calculate total self-concept score
    dimension_columns = list(SELF_CONCEPT_DIMENSIONS.keys())
    available_dimensions = [col for col in dimension_columns if col in self_concept_data.columns]
    if available_dimensions:
        self_concept_data['Total_Self_Concept'] = self_concept_data[available_dimensions].mean(axis=1)
    
    print(f"Self-Concept data processed: {len(self_concept_data)} participants")
    print(f"Dimensions calculated: {[col for col in self_concept_data.columns if col != 'participant']}")
    
    return self_concept_data

def merge_datasets():
    """Merge goal orientation and self-concept datasets"""
    print("Merging datasets...")
    
    # Process both datasets
    goal_data = process_goal_orientation_data()
    self_concept_data = process_self_concept_data()
    
    # Convert participant columns to string for consistent merging
    goal_data['participant'] = goal_data['participant'].astype(str)
    self_concept_data['participant'] = self_concept_data['participant'].astype(str)
    
    print(f"Goal Orientation participants: {goal_data['participant'].tolist()}")
    print(f"Self-Concept participants: {self_concept_data['participant'].tolist()}")
    
    # Merge on participant
    combined_data = pd.merge(goal_data, self_concept_data, on='participant', how='inner')
    
    print(f"Combined data: {len(combined_data)} participants with both Goal Orientation and Self-Concept data")
    print(f"Participants: {combined_data['participant'].tolist()}")
    
    # Save combined dataset
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\combined_goal_self_concept_data.csv'
    combined_data.to_csv(output_path, index=False)
    print(f"Combined dataset saved to: {output_path}")
    
    # Display basic statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Total participants: {len(combined_data)}")
    
    goal_dims = ['Performance_Approach', 'Performance_Avoidance', 'Mastery_Approach', 'Mastery_Avoidance']
    self_concept_dims = list(SELF_CONCEPT_DIMENSIONS.keys()) + ['Total_Self_Concept']
    
    print(f"Goal Orientation dimensions: {', '.join(goal_dims)}")
    print(f"Self-Concept dimensions: {', '.join([dim.replace('_', '/') for dim in SELF_CONCEPT_DIMENSIONS.keys()])}")
    
    # Basic statistics
    print("\n=== GOAL ORIENTATION SCORE SUMMARY ===")
    goal_columns = [col for col in combined_data.columns if col in goal_dims]
    if goal_columns:
        print(combined_data[goal_columns].describe())
    
    print("\n=== SELF-CONCEPT SCORE SUMMARY ===")
    sc_columns = [col for col in combined_data.columns if col in self_concept_dims]
    if sc_columns:
        print(combined_data[sc_columns].describe())
    
    return combined_data

def main():
    """Main preprocessing function"""
    print("GOAL ORIENTATION - SELF CONCEPT DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        combined_data = merge_datasets()
        
        print("\nData preprocessing completed successfully!")
        print("Next steps: Run statistical analysis and visualization scripts.")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    main()