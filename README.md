# Psychometric Analysis Project

A comprehensive Python-based psychometric assessment and analysis system that integrates multiple psychological instruments to provide detailed individual profiles and insights.

## 📋 Overview

This project analyzes three core psychological dimensions:

1. **RIASEC Career Interests** - Based on Holland's Theory of Career Choice
2. **Achievement Goal Orientations** - Performance vs. Mastery approaches
3. **Multidimensional Self-Concept** - Self-perception across multiple domains

The system processes survey responses, performs statistical analysis, generates visualizations, and creates comprehensive individual and group reports.

## 🎯 Key Features

- **Multi-dimensional Assessment Integration**: Combines RIASEC, goal orientation, and self-concept data
- **Statistical Analysis**: Descriptive statistics, correlations, clustering, and pattern identification
- **Individual Profiling**: Comprehensive psychological profiles for each participant
- **Automated Report Generation**: PDF reports with visualizations and interpretations
- **Career Matching**: RIASEC-based career recommendations using O*NET database
- **Cross-Domain Analysis**: Examines relationships between different psychological constructs

## 📁 Project Structure

```
.
├── RIASEC/                          # RIASEC Career Interest Analysis
│   ├── riasec_analysis.py           # Core RIASEC statistical analysis
│   ├── authentic_onet_profiler.py   # Career matching with O*NET integration
│   ├── create_pdf_report.py         # RIASEC report generation
│   ├── responses.csv                # RIASEC survey responses
│   └── career.csv                   # Career database with RIASEC codes
│
├── goal_orientation/                # Achievement Goal Orientation Analysis
│   ├── goal_orientation_analysis.py # Goal orientation scoring and categorization
│   └── responses.csv                # Goal orientation survey responses
│
├── self_concept/                    # Multidimensional Self-Concept Analysis
│   ├── self_concept_analysis.py     # Self-concept statistical analysis
│   ├── create_pdf_report.py         # Self-concept report generation
│   └── responses.csv                # Self-concept survey responses
│
├── goal_riasec_analysis/            # Integrated Goal-RIASEC Analysis
│   ├── data_preprocessing.py        # Data integration and cleaning
│   ├── statistical_analysis.py      # Correlation and statistical tests
│   ├── create_visualizations.py     # Visualization generation
│   └── create_pdf_reports.py        # Integrated report generation
│
├── goal_self_concept_analysis/      # Integrated Goal-Self Concept Analysis
│   ├── data_preprocessing.py        # Data integration and cleaning
│   ├── statistical_analysis.py      # Correlation and clustering analysis
│   └── create_visualizations.py     # Visualization generation
│
├── riasec_self_concept_analysis/    # Integrated RIASEC-Self Concept Analysis
│   └── [Analysis scripts]           # Cross-domain analysis
│
└── individual_analysis/             # Comprehensive Individual Profiles
    ├── comprehensive_individual_analysis.py  # Multi-domain profile generation
    ├── create_individual_pdf_report.py       # Individual report generation
    └── Individual_Analysis_Report.md         # Summary report
```

## 🔬 Assessment Instruments

### 1. RIASEC Interest Inventory

Based on **Holland's Theory of Career Choice**, measuring six interest types:

- **R**ealistic: Hands-on, practical, mechanical activities
- **I**nvestigative: Analytical, scientific, problem-solving
- **A**rtistic: Creative, expressive, imaginative activities
- **S**ocial: Helping, teaching, counseling others
- **E**nterprising: Leading, persuading, entrepreneurial activities
- **C**onventional: Organizing, detail-oriented, structured tasks

**Output**: Holland Code (3-letter code representing top interests)

### 2. Achievement Goal Orientation

Measures four goal orientation types:

- **Performance-Approach**: Focus on outperforming others
- **Mastery-Approach**: Focus on learning and skill development
- **Performance-Avoidance**: Avoiding poor performance relative to others
- **Mastery-Avoidance**: Worry about not learning enough

**Output**: Categorical classification (e.g., "Mastery-Focused Learner", "Performance-Anxious Avoider")

### 3. Multidimensional Self-Concept

49-item Likert scale assessment measuring self-perception across domains:

- Academic self-concept
- Social self-concept
- Physical self-concept
- Emotional self-concept
- General self-worth

**Output**: Overall self-concept score and categorical level (Poor/Moderate/Good/Excellent)

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scipy reportlab pillow requests
```

### Basic Usage

#### 1. Run Individual Analysis

```bash
cd RIASEC
python riasec_analysis.py
```

#### 2. Generate Comprehensive Reports

```bash
cd individual_analysis
python comprehensive_individual_analysis.py
python create_individual_pdf_report.py
```

#### 3. Run Cross-Domain Analysis

```bash
cd goal_riasec_analysis
python data_preprocessing.py
python statistical_analysis.py
python create_visualizations.py
python create_pdf_reports.py
```

## 📊 Analysis Capabilities

### Descriptive Statistics
- Mean, median, standard deviation for all scales
- Distribution analysis and normality tests
- Frequency distributions and percentile rankings

### Inferential Statistics
- Correlation analysis (Pearson, Spearman)
- Statistical significance testing
- Pattern identification and clustering

### Visualization
- Box plots and violin plots
- Heatmaps and correlation matrices
- Individual profile radar charts
- Distribution histograms

### Report Generation
- PDF reports with professional formatting
- Individual participant profiles
- Group-level statistical summaries
- Career recommendations and insights

## 📈 Output Files

Each analysis module generates:

- **CSV Files**: Processed data, scores, and statistics
- **PNG Files**: Visualizations and charts
- **PDF Reports**: Comprehensive formatted reports
- **TXT/MD Files**: Summary reports and documentation

## 🎓 Use Cases

- **Career Counseling**: Match individuals to suitable career paths
- **Educational Assessment**: Understand student motivation and self-perception
- **Research**: Examine relationships between personality, goals, and self-concept
- **Personal Development**: Self-awareness and growth insights
- **Team Building**: Understanding team member strengths and preferences

## 📚 Theoretical Background

### Holland's RIASEC Theory
Developed by John Holland (1959), proposes that people and work environments can be classified into six types. Career satisfaction and success are highest when personality type matches work environment.

### Achievement Goal Theory
Based on Elliot and McGregor's (2001) 2x2 achievement goal framework, distinguishing between mastery vs. performance goals and approach vs. avoidance orientations.

### Self-Concept Theory
Based on Marsh's multidimensional hierarchical model of self-concept, recognizing that self-perception varies across different domains of life.

## 🔧 Customization

### Adding New Participants
1. Add responses to respective `responses.csv` files
2. Run the analysis scripts
3. New profiles will be automatically generated

### Modifying Career Database
Edit `RIASEC/career.csv` to add/modify career options and their RIASEC codes.

### Adjusting Statistical Parameters
Modify thresholds and parameters in individual analysis scripts.

## 📝 Data Format

All survey data should be in CSV format with:
- First row: Headers/questions
- First column: Timestamp (optional)
- Last column: Participant name
- Middle columns: Survey responses

## 🤝 Contributing

This is an academic/research project. For modifications:
1. Maintain the theoretical integrity of assessments
2. Document any changes to scoring algorithms
3. Validate statistical procedures
4. Test report generation after changes

## 📄 License

This project is for educational and research purposes.

## 🙋 Support

For questions about:
- **Assessment interpretation**: Refer to respective theory documentation
- **Technical issues**: Check error logs and data format
- **Statistical methods**: See scipy and pandas documentation

## 📖 References

- Holland, J. L. (1997). Making vocational choices: A theory of vocational personalities and work environments.
- Elliot, A. J., & McGregor, H. A. (2001). A 2× 2 achievement goal framework. Journal of Personality and Social Psychology, 80(3), 501.
- Marsh, H. W. (1990). The structure of academic self-concept: The Marsh/Shavelson model. Journal of Educational Psychology, 82(4), 623.

---

**Last Updated**: January 2026  
**Version**: 1.0
