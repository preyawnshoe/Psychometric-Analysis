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
        'Realistic': [1, 7, 14, 21, 22, 30, 32, 37],  # Hands-on, practical
        'Investigative': [2, 11, 18, 26, 39],  # Scientific, analytical
        'Artistic': [8, 17, 23, 27, 31, 41],  # Creative, expressive
        'Social': [12, 13, 20, 28, 40],  # Helping, teaching
        'Enterprising': [5, 10, 16, 19, 29, 34, 36, 42],  # Leading, persuading
        'Conventional': [3, 6, 9, 15, 24, 25, 33, 35, 38]  # Organizing, detail-oriented
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

def process_self_concept_data():
    """Process self-concept survey data based on the mapping in table.md"""
    print("Processing Self-Concept data...")
    
    # Read self-concept data
    self_concept_df = pd.read_csv(r'C:\Users\p2var\vandana\self_concept\responses.csv')
    
    # Clean participant names - the Name column is at index 1
    self_concept_df['participant'] = self_concept_df['Name'].apply(clean_participant_name)
    
    # Remove rows without participant names
    self_concept_df = self_concept_df.dropna(subset=['participant'])
    
    # Self-concept dimension mapping based on table.md
    # Questions are in columns starting from index 8 (after demographic info)
    self_concept_mapping = {
        'Health_Sex_Appropriateness': {
            'questions': [6, 32, 34, 46],
            'polarities': ['P', 'P', 'N', 'P']
        },
        'Abilities': {
            'questions': [4, 8, 12, 22, 34, 36, 37, 40],
            'polarities': ['P', 'P', 'N', 'N', 'P', 'N', 'N', 'P']
        },
        'Self_Confidence': {
            'questions': [7, 9, 14, 16, 42],
            'polarities': ['P', 'P', 'N', 'N', 'P']
        },
        'Self_Acceptance': {
            'questions': [2, 10, 17, 33],
            'polarities': ['P', 'N', 'N', 'N']
        },
        'Worthiness': {
            'questions': [1, 3, 19, 24, 39, 46],  # Removed duplicate 24
            'polarities': ['P', 'N', 'N', 'P', 'N', 'P']
        },
        'Present_Past_Future': {
            'questions': [18, 21, 25, 26, 40],  # Adjusted after removing 20, 29
            'polarities': ['P', 'P', 'P', 'P', 'P']
        },
        'Beliefs_Convictions': {
            'questions': [23, 45, 47],
            'polarities': ['N', 'P', 'P']
        },
        'Shame_Guilt': {
            'questions': [5, 13, 27, 28, 48],
            'polarities': ['N', 'N', 'N', 'N', 'N']
        },
        'Sociability': {
            'questions': [31, 35, 41, 43],
            'polarities': ['P', 'P', 'N', 'P']
        },
        'Emotional': {
            'questions': [11, 15, 20, 49],  # Adjusted for question 20 removal
            'polarities': ['N', 'N', 'N', 'N']
        }
    }
    
    # Get question columns - they start from column 8 onwards
    question_columns = self_concept_df.columns[8:].tolist()  # Skip timestamp, name, demographics
    print(f"Found {len(question_columns)} question columns")
    
    # Convert responses to numeric
    response_mapping = {
        'Strongly agree': 5, 'Agree': 4, 'Undecided': 3, 'Disagree': 2, 'Strongly disagree': 1,
        'strongly agree': 5, 'agree': 4, 'undecided': 3, 'disagree': 2, 'strongly disagree': 1,
        'Strongly Agree': 5, 'Strongly Disagree': 1,
        5: 5, 4: 4, 3: 3, 2: 2, 1: 1,
        '5': 5, '4': 4, '3': 3, '2': 2, '1': 1
    }
    
    # Apply response mapping to question columns only (exclude participant and demographic columns)
    for col in question_columns:
        if col in self_concept_df.columns and col != 'participant':
            self_concept_df[col] = self_concept_df[col].map(response_mapping).fillna(self_concept_df[col])
            # Try to convert to numeric if still strings
            self_concept_df[col] = pd.to_numeric(self_concept_df[col], errors='coerce')
    
    # Calculate self-concept dimension scores
    # Create a list to store participant data
    processed_participants = []
    
    for participant_idx, participant_row in self_concept_df.iterrows():
        participant_name = participant_row['participant']
        participant_data = {'participant': participant_name}
        
        # Calculate each dimension score for this participant
        for dimension, config in self_concept_mapping.items():
            questions = config['questions']
            polarities = config['polarities']
            
            scores = []
            for i, q_num in enumerate(questions):
                # Question number q_num corresponds to column index (q_num - 1 + 8) 
                # because questions start at column 8 and are 1-indexed
                if q_num <= len(question_columns):
                    col_idx = q_num - 1 + 8  # Adjust for 0-indexing and offset
                    if col_idx < len(self_concept_df.columns):
                        q_col = self_concept_df.columns[col_idx]
                        raw_score = participant_row[q_col]
                        
                        if pd.notna(raw_score) and isinstance(raw_score, (int, float)):
                            # Apply polarity
                            if polarities[i] == 'N':  # Negative item - reverse score
                                processed_score = 6 - raw_score  # Reverse 5-point scale
                            else:  # Positive item
                                processed_score = raw_score
                            scores.append(processed_score)
                
            # Calculate mean score for this dimension
            if scores:
                participant_data[dimension] = np.mean(scores)
            else:
                participant_data[dimension] = np.nan
        
        processed_participants.append(participant_data)
    
    # Create new DataFrame from processed data
    self_concept_processed = pd.DataFrame(processed_participants)
    
    # Calculate total self-concept score
    dimension_columns = list(self_concept_mapping.keys())
    available_dimensions = [col for col in dimension_columns if col in self_concept_processed.columns]
    if available_dimensions:
        self_concept_processed['Total_Self_Concept'] = self_concept_processed[available_dimensions].mean(axis=1)
    
    # Output columns
    output_columns = ['participant'] + available_dimensions + (['Total_Self_Concept'] if 'Total_Self_Concept' in self_concept_processed.columns else [])
    
    print(f"Self-Concept data processed: {len(self_concept_processed)} participants")
    print(f"Dimensions calculated: {available_dimensions}")
    return self_concept_processed

def merge_datasets():
    """Merge RIASEC and self-concept datasets"""
    print("Merging datasets...")
    
    # Process both datasets
    riasec_data = process_riasec_data()
    self_concept_data = process_self_concept_data()
    
    # Check participant columns data types and values for debugging
    print(f"RIASEC participants: {len(riasec_data)} total")
    print(f"Self-Concept participants: {len(self_concept_data)} total")
    
    # Convert participant columns to string for consistent merging
    riasec_data['participant'] = riasec_data['participant'].astype(str)
    self_concept_data['participant'] = self_concept_data['participant'].astype(str)
    
    # Merge on participant
    combined_data = pd.merge(riasec_data, self_concept_data, on='participant', how='inner')
    
    print(f"Combined data: {len(combined_data)} participants with both RIASEC and Self-Concept data")
    print(f"Participants: {combined_data['participant'].tolist()}")
    
    # Save combined dataset
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\combined_riasec_self_concept_data.csv'
    combined_data.to_csv(output_path, index=False)
    print(f"Combined dataset saved to: {output_path}")
    
    # Display basic statistics
    print("\n=== DATASET SUMMARY ===")
    print(f"Total participants: {len(combined_data)}")
    print(f"RIASEC dimensions: Realistic, Investigative, Artistic, Social, Enterprising, Conventional")
    print(f"Self-Concept dimensions: Health/Sex Appropriateness, Abilities, Self-Confidence, Self-Acceptance, Worthiness, Present/Past/Future, Beliefs/Convictions, Shame/Guilt, Sociability, Emotional")
    
    print("\n=== RIASEC SCORE SUMMARY ===")
    riasec_cols = ['Realistic', 'Investigative', 'Artistic', 'Social', 'Enterprising', 'Conventional']
    print(combined_data[riasec_cols].describe().round(3))
    
    print("\n=== SELF-CONCEPT SCORE SUMMARY ===")
    self_concept_cols = ['Health_Sex_Appropriateness', 'Abilities', 'Self_Confidence', 'Self_Acceptance', 
                        'Worthiness', 'Present_Past_Future', 'Beliefs_Convictions', 'Shame_Guilt', 
                        'Sociability', 'Emotional']
    available_cols = [col for col in self_concept_cols if col in combined_data.columns]
    if available_cols:
        print(combined_data[available_cols].describe().round(3))
    
    return combined_data

if __name__ == "__main__":
    # Set up the analysis environment
    os.chdir(r'C:\Users\p2var\vandana\riasec_self_concept_analysis')
    
    # Process and merge data
    combined_data = merge_datasets()
    
    print("\nData preprocessing completed successfully!")
    print("Next steps: Run correlation analysis and visualization scripts.")