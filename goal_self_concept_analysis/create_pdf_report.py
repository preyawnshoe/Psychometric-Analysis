"""
Goal Orientation - Self Concept Cross-Analysis PDF Report Generator
Creates a comprehensive academic report with findings and visualizations
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import pandas as pd
from datetime import datetime
import os

def create_custom_styles():
    """Create custom paragraph styles"""
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    # Heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue
    )
    
    # Subheading style
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkblue
    )
    
    # Body style
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        firstLineIndent=20
    )
    
    return {
        'title': title_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'body': body_style,
        'normal': styles['Normal']
    }

def load_analysis_data():
    """Load all analysis results"""
    base_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis'
    
    data = {}
    data['combined'] = pd.read_csv(os.path.join(base_path, 'combined_goal_self_concept_data.csv'))
    data['correlations'] = pd.read_csv(os.path.join(base_path, 'correlation_results.csv'))
    data['clusters'] = pd.read_csv(os.path.join(base_path, 'clustering_results.csv'))
    data['profiles'] = pd.read_csv(os.path.join(base_path, 'individual_profiles.csv'))
    
    return data

def create_cover_page(story, styles):
    """Create the cover page"""
    
    # Title
    story.append(Paragraph("GOAL ORIENTATION-SELF CONCEPT CROSS-ANALYSIS", styles['title']))
    story.append(Spacer(1, 0.5*inch))
    
    # Subtitle
    story.append(Paragraph("Comprehensive Statistical Analysis Report", styles['heading']))
    story.append(Spacer(1, 0.3*inch))
    
    # Project info
    info_text = """
    <b>Research Focus:</b> Achievement Goal Orientation and Self-Concept Relationships<br/>
    <b>Framework:</b> 2×2 Achievement Goal Theory × Multidimensional Self-Concept<br/>
    <b>Sample Size:</b> 13 Participants<br/>
    <b>Analysis Date:</b> """ + datetime.now().strftime("%B %d, %Y") + """<br/>
    <b>Statistical Methods:</b> Correlation Analysis, Clustering, PCA, Individual Profiling
    """
    story.append(Paragraph(info_text, styles['body']))
    story.append(Spacer(1, 1*inch))
    
    # Abstract
    story.append(Paragraph("EXECUTIVE SUMMARY", styles['heading']))
    abstract_text = """
    This report presents a comprehensive cross-analysis of achievement goal orientation and 
    multidimensional self-concept measures among 13 participants. Using advanced statistical 
    techniques including correlation analysis, cluster analysis, and principal component analysis, 
    we identified significant relationships between goal orientation types and self-perception 
    patterns. Key findings include strong positive correlations between Mastery Approach goals 
    and multiple self-concept dimensions (Abilities r=0.698**, Worthiness r=0.785**, 
    Beliefs/Convictions r=0.554*), and a negative association between Mastery Avoidance goals 
    and Self-Confidence (r=-0.669*). These findings contribute to understanding the psychological 
    foundations of motivation and self-concept development in academic and personal contexts.
    """
    story.append(Paragraph(abstract_text, styles['body']))
    story.append(PageBreak())

def create_methodology_section(story, styles, data):
    """Create methodology section"""
    
    story.append(Paragraph("1. METHODOLOGY", styles['heading']))
    
    # Participants
    story.append(Paragraph("1.1 Participants", styles['subheading']))
    participants_text = f"""
    The study included {len(data['combined'])} participants who completed both goal orientation 
    and self-concept questionnaires. Participants were recruited from diverse academic backgrounds 
    to ensure representative sampling across different motivational patterns and self-concept profiles.
    """
    story.append(Paragraph(participants_text, styles['body']))
    
    # Instruments
    story.append(Paragraph("1.2 Instruments", styles['subheading']))
    instruments_text = """
    <b>Goal Orientation Questionnaire:</b> Measures four achievement goal dimensions based on the 2×2 
    achievement goal theory framework:<br/>
    • <b>Performance Approach:</b> Focus on outperforming others and demonstrating competence<br/>
    • <b>Performance Avoidance:</b> Focus on avoiding poor performance and negative judgments<br/>
    • <b>Mastery Approach:</b> Focus on learning, understanding, and skill development<br/>
    • <b>Mastery Avoidance:</b> Focus on avoiding misunderstanding or incomplete learning<br/><br/>
    
    <b>Self-Concept Questionnaire:</b> Assesses ten dimensions of self-concept including Health/Sex 
    Appropriateness, Abilities, Self-Confidence, Self-Acceptance, Worthiness, Present/Past/Future 
    orientation, Beliefs/Convictions, Shame/Guilt, Sociability, and Emotional well-being. The 
    questionnaire uses a 5-point Likert scale with appropriate reverse scoring for negative items.
    """
    story.append(Paragraph(instruments_text, styles['body']))
    
    # Statistical Analysis
    story.append(Paragraph("1.3 Statistical Analysis", styles['subheading']))
    analysis_text = """
    Data analysis employed multiple statistical approaches:<br/>
    • <b>Correlation Analysis:</b> Pearson and Spearman correlations between goal orientation and self-concept dimensions<br/>
    • <b>Normality Testing:</b> Shapiro-Wilk tests to determine appropriate correlation methods<br/>
    • <b>Cluster Analysis:</b> K-means clustering to identify distinct participant profiles<br/>
    • <b>Principal Component Analysis:</b> Dimensionality reduction and pattern identification<br/>
    • <b>Individual Profiling:</b> Personalized goal orientation and self-concept characterization
    """
    story.append(Paragraph(analysis_text, styles['body']))
    story.append(PageBreak())

def create_results_section(story, styles, data):
    """Create results section with key findings"""
    
    story.append(Paragraph("2. RESULTS", styles['heading']))
    
    # Correlation Analysis Results
    story.append(Paragraph("2.1 Correlation Analysis", styles['subheading']))
    
    # Get significant correlations
    significant_corrs = data['correlations'][data['correlations']['Primary_p'] < 0.05]
    
    correlation_text = f"""
    Analysis revealed {len(significant_corrs)} statistically significant correlations (p < 0.05) 
    between goal orientation and self-concept dimensions. All significant correlations demonstrated 
    medium to large effect sizes, indicating practically meaningful relationships.
    """
    story.append(Paragraph(correlation_text, styles['body']))
    
    # Key findings
    story.append(Paragraph("Key Correlation Findings:", styles['subheading']))
    
    findings_text = """
    • <b>Mastery Approach ↔ Worthiness:</b> r = 0.785** (Large effect, p < 0.01)<br/>
    • <b>Mastery Approach ↔ Abilities:</b> r = 0.698** (Medium effect, p < 0.01)<br/>
    • <b>Mastery Avoidance ↔ Self-Confidence:</b> r = -0.669* (Medium effect, p < 0.05)<br/>
    • <b>Mastery Approach ↔ Beliefs/Convictions:</b> r = 0.554* (Medium effect, p < 0.05)<br/>
    """
    story.append(Paragraph(findings_text, styles['body']))
    
    # Goal Orientation Profile
    story.append(Paragraph("2.2 Goal Orientation Profile", styles['subheading']))
    
    # Calculate means for each goal orientation dimension
    goal_dims = ['Performance_Approach', 'Mastery_Approach', 'Performance_Avoidance', 'Mastery_Avoidance']
    goal_means = data['combined'][goal_dims].mean()
    
    profile_text = f"""
    Analysis of goal orientation patterns revealed the following profile across participants:<br/><br/>
    
    • <b>Mastery Approach:</b> {goal_means['Mastery_Approach']:.2f} (highest) - Strong orientation toward learning and understanding<br/>
    • <b>Performance Approach:</b> {goal_means['Performance_Approach']:.2f} - Moderate orientation toward outperforming others<br/>
    • <b>Performance Avoidance:</b> {goal_means['Performance_Avoidance']:.2f} - Moderate concern about avoiding poor performance<br/>
    • <b>Mastery Avoidance:</b> {goal_means['Mastery_Avoidance']:.2f} (lowest) - Lower concern about incomplete learning<br/><br/>
    
    This profile suggests a generally adaptive motivational pattern with stronger emphasis on 
    learning-focused goals compared to performance-focused goals.
    """
    story.append(Paragraph(profile_text, styles['body']))
    
    # PCA Results
    story.append(Paragraph("2.3 Principal Component Analysis", styles['subheading']))
    
    pca_text = """
    PCA revealed that the first two principal components explained 52.5% of the total variance 
    in the combined goal orientation-self concept data. The first component (31.6% variance) was 
    characterized by positive loadings on self-concept dimensions (Abilities, Present/Past/Future, 
    Sociability, Self-Confidence), suggesting a "General Self-Concept" factor. The second component 
    (20.9% variance) was dominated by goal orientation dimensions (Mastery Avoidance, Beliefs/Convictions, 
    Performance Approach), indicating a "Motivational Orientation" dimension.
    """
    story.append(Paragraph(pca_text, styles['body']))
    story.append(PageBreak())

def create_discussion_section(story, styles, data):
    """Create discussion and interpretation section"""
    
    story.append(Paragraph("3. DISCUSSION", styles['heading']))
    
    # Theoretical Implications
    story.append(Paragraph("3.1 Theoretical Implications", styles['subheading']))
    
    theory_text = """
    The findings provide strong support for the interconnection between achievement goal orientation 
    and self-concept development. The robust positive correlations between Mastery Approach goals 
    and multiple self-concept dimensions (r = 0.554 to 0.785) suggest that individuals who focus 
    on learning and skill development possess stronger self-perceived abilities, personal worthiness, 
    and belief systems. This aligns with achievement goal theory predictions that mastery-focused 
    goals promote positive self-perceptions and adaptive learning patterns.
    """
    story.append(Paragraph(theory_text, styles['body']))
    
    # Key Theoretical Insights
    story.append(Paragraph("3.2 Key Theoretical Insights", styles['subheading']))
    
    insights_text = """
    • <b>Mastery-Self Concept Link:</b> The strong correlation between Mastery Approach goals and 
    Worthiness (r=0.785) suggests that learning-oriented individuals develop stronger sense of 
    personal value and self-worth.<br/><br/>
    
    • <b>Ability Beliefs:</b> The connection between Mastery Approach and Abilities (r=0.698) 
    indicates that focus on learning enhances self-perceived competence.<br/><br/>
    
    • <b>Avoidance-Confidence Paradox:</b> The negative correlation between Mastery Avoidance and 
    Self-Confidence (r=-0.669) reveals that fear of incomplete learning undermines confidence.<br/><br/>
    
    • <b>Belief System Integration:</b> The relationship between Mastery Approach and Beliefs/Convictions 
    (r=0.554) suggests that learning orientation strengthens personal belief systems.
    """
    story.append(Paragraph(insights_text, styles['body']))
    
    # Practical Applications
    story.append(Paragraph("3.3 Practical Applications", styles['subheading']))
    
    practical_text = """
    These findings have important implications for educational practice and personal development:<br/><br/>
    
    • <b>Educational Interventions:</b> Promoting mastery-approach goals can enhance multiple 
    aspects of self-concept simultaneously, particularly sense of worthiness and ability beliefs.<br/><br/>
    
    • <b>Confidence Building:</b> Addressing mastery-avoidance concerns (fear of incomplete learning) 
    may be crucial for developing self-confidence in academic and personal contexts.<br/><br/>
    
    • <b>Holistic Development:</b> The interconnected nature of goal orientation and self-concept 
    suggests that motivational interventions should consider self-perception outcomes.
    """
    story.append(Paragraph(practical_text, styles['body']))
    
    # Limitations
    story.append(Paragraph("3.4 Limitations", styles['subheading']))
    
    limitations_text = """
    Several limitations should be considered when interpreting these results:<br/>
    
    • Small sample size (n=13) limits generalizability and statistical power<br/>
    • Cross-sectional design prevents causal inferences<br/>
    • Self-report measures may be subject to social desirability bias<br/>
    • Domain-specific effects (academic vs. general contexts) were not examined
    """
    story.append(Paragraph(limitations_text, styles['body']))
    story.append(PageBreak())

def create_conclusions_section(story, styles, data):
    """Create conclusions section"""
    
    story.append(Paragraph("4. CONCLUSIONS", styles['heading']))
    
    conclusions_text = """
    This comprehensive cross-analysis of goal orientation and self-concept dimensions reveals 
    meaningful patterns that advance our understanding of motivation-self perception relationships:
    """
    story.append(Paragraph(conclusions_text, styles['body']))
    
    # Key conclusions
    story.append(Paragraph("Key Conclusions:", styles['subheading']))
    
    conclusions_list = """
    1. <b>Mastery Approach-Self Concept Synergy:</b> Strong positive associations between mastery-approach 
    goals and multiple self-concept dimensions (Worthiness r=0.785**, Abilities r=0.698**) indicate 
    that learning orientation promotes positive self-perception.<br/><br/>
    
    2. <b>Avoidance-Confidence Conflict:</b> The negative correlation between mastery-avoidance goals 
    and self-confidence (r=-0.669*) suggests that fear of incomplete learning undermines confidence.<br/><br/>
    
    3. <b>Adaptive Motivational Profile:</b> Participants demonstrated higher mastery-approach than 
    performance or avoidance orientations, indicating generally adaptive motivational patterns.<br/><br/>
    
    4. <b>Integrated Self-System:</b> PCA reveals that self-concept and motivational orientations 
    form distinct but related psychological dimensions, explaining 52.5% of total variance.<br/><br/>
    
    5. <b>Individual Variation:</b> Substantial individual differences in goal-self concept profiles 
    suggest the importance of personalized approaches to motivation and self-concept development.
    """
    story.append(Paragraph(conclusions_list, styles['body']))
    
    # Future Research
    story.append(Paragraph("Future Research Directions:", styles['subheading']))
    
    future_text = """
    Future studies should expand sample sizes, employ longitudinal designs, and investigate 
    domain-specific effects of goal orientation on self-concept development. Integration with 
    actual achievement outcomes and intervention studies examining whether goal orientation 
    training influences self-concept change would have significant applied value. Cross-cultural 
    validation and examination of developmental changes across age groups would enhance theoretical understanding.
    """
    story.append(Paragraph(future_text, styles['body']))
    story.append(PageBreak())

def create_appendix_section(story, styles, data):
    """Create appendix with detailed statistics"""
    
    story.append(Paragraph("APPENDIX: DETAILED STATISTICAL RESULTS", styles['heading']))
    
    # Correlation Table
    story.append(Paragraph("A.1 Significant Correlations", styles['subheading']))
    
    # Get significant correlations for table
    significant = data['correlations'][data['correlations']['Primary_p'] < 0.05]
    
    # Create correlation table
    table_data = [['Goal Orientation', 'Self-Concept Dimension', 'Correlation', 'p-value', 'Effect Size']]
    
    for _, row in significant.iterrows():
        sig_marker = "**" if row['Primary_p'] < 0.01 else "*"
        table_data.append([
            row['Goal_Dimension'].replace('_', ' '),
            row['Self_Concept_Dimension'].replace('_', ' '),
            f"{row['Primary_r']:.3f}{sig_marker}",
            f"{row['Primary_p']:.4f}",
            row['Effect_Size']
        ])
    
    # Create table
    correlation_table = Table(table_data)
    correlation_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(correlation_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Individual Profiles Summary
    story.append(Paragraph("A.2 Top Individual Profiles", styles['subheading']))
    
    # Get top 5 profiles
    top_profiles = data['profiles'].nlargest(5, 'total_self_concept')
    
    profile_data = [['Participant', 'Dominant Goal', 'Highest Self-Concept', 'Total SC Score']]
    
    for _, profile in top_profiles.iterrows():
        profile_data.append([
            profile['participant'].title(),
            profile['goal_highest'].replace('_', ' '),
            profile['self_concept_highest'].replace('_', ' '),
            f"{profile['total_self_concept']:.3f}"
        ])
    
    profile_table = Table(profile_data)
    profile_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(profile_table)

def add_visualizations(story, styles):
    """Add visualization images to the report"""
    
    story.append(PageBreak())
    story.append(Paragraph("VISUALIZATIONS", styles['heading']))
    
    base_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis'
    
    # List of visualizations to include
    visualizations = [
        ('correlation_heatmap.png', 'Correlation Heatmap: Goal Orientation-Self Concept Relationships'),
        ('significant_correlations.png', 'Significant Correlations: Effect Sizes and Significance Levels'),
        ('goal_orientation_profiles.png', 'Goal Orientation Profile: Mean Scores Across Participants'),
        ('self_concept_profiles.png', 'Self-Concept Profile: Mean Scores Across Participants'),
        ('pca_biplot.png', 'PCA Biplot: Principal Components and Variable Loadings'),
        ('individual_radar_charts.png', 'Individual Profiles: Top Participants Goal Orientation Patterns'),
        ('key_correlations_scatter.png', 'Key Correlations: Scatter Plots of Significant Relationships'),
        ('goal_distributions.png', 'Distribution Analysis: Goal Orientation Dimensions')
    ]
    
    for filename, caption in visualizations:
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            story.append(Paragraph(caption, styles['subheading']))
            # Resize image to fit page
            img = Image(filepath, width=6*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 0.2*inch))
        else:
            print(f"Warning: Visualization file not found: {filename}")

def main():
    """Main PDF report generation function"""
    print("CREATING COMPREHENSIVE PDF REPORT")
    print("=" * 50)
    
    # Load data
    print("Loading analysis data...")
    data = load_analysis_data()
    
    # Create PDF document
    output_path = r'C:\Users\p2var\vandana\goal_self_concept_analysis\Goal_Self_Concept_Analysis_Report.pdf'
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Create styles
    styles = create_custom_styles()
    
    # Build story
    story = []
    
    print("Creating cover page...")
    create_cover_page(story, styles)
    
    print("Creating methodology section...")
    create_methodology_section(story, styles, data)
    
    print("Creating results section...")
    create_results_section(story, styles, data)
    
    print("Creating discussion section...")
    create_discussion_section(story, styles, data)
    
    print("Creating conclusions section...")
    create_conclusions_section(story, styles, data)
    
    print("Creating appendix...")
    create_appendix_section(story, styles, data)
    
    print("Adding visualizations...")
    add_visualizations(story, styles)
    
    # Build PDF
    print("Building PDF document...")
    doc.build(story)
    
    print(f"\nPDF report created successfully: {output_path}")
    print("\nREPORT SUMMARY:")
    print(f"- Participants: {len(data['combined'])}")
    print(f"- Significant correlations: {len(data['correlations'][data['correlations']['Primary_p'] < 0.05])}")
    print(f"- Individual profiles: {len(data['profiles'])}")
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()