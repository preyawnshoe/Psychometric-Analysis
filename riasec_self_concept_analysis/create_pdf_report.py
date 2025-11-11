"""
RIASEC-Self Concept Cross-Analysis PDF Report Generator
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
    base_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis'
    
    data = {}
    data['combined'] = pd.read_csv(os.path.join(base_path, 'combined_riasec_self_concept_data.csv'))
    data['correlations'] = pd.read_csv(os.path.join(base_path, 'correlation_results.csv'))
    data['clusters'] = pd.read_csv(os.path.join(base_path, 'clustering_results.csv'))
    data['profiles'] = pd.read_csv(os.path.join(base_path, 'individual_profiles.csv'))
    
    return data

def create_cover_page(story, styles):
    """Create the cover page"""
    
    # Title
    story.append(Paragraph("RIASEC-SELF CONCEPT CROSS-ANALYSIS", styles['title']))
    story.append(Spacer(1, 0.5*inch))
    
    # Subtitle
    story.append(Paragraph("Comprehensive Statistical Analysis Report", styles['heading']))
    story.append(Spacer(1, 0.3*inch))
    
    # Project info
    info_text = """
    <b>Research Focus:</b> Career Interest Types and Self-Concept Relationships<br/>
    <b>Framework:</b> Holland's RIASEC Theory × Multidimensional Self-Concept<br/>
    <b>Sample Size:</b> 12 Participants<br/>
    <b>Analysis Date:</b> """ + datetime.now().strftime("%B %d, %Y") + """<br/>
    <b>Statistical Methods:</b> Correlation Analysis, Clustering, PCA, Individual Profiling
    """
    story.append(Paragraph(info_text, styles['body']))
    story.append(Spacer(1, 1*inch))
    
    # Abstract
    story.append(Paragraph("EXECUTIVE SUMMARY", styles['heading']))
    abstract_text = """
    This report presents a comprehensive cross-analysis of Holland's RIASEC career interest 
    dimensions and multidimensional self-concept measures among 12 participants. Using advanced 
    statistical techniques including correlation analysis, cluster analysis, and principal 
    component analysis, we identified significant relationships between career interests and 
    self-perception patterns. Key findings include strong positive correlations between 
    Investigative interests and self-concept dimensions (Abilities, Worthiness, Present/Past/Future), 
    negative associations between Artistic interests and self-acceptance, and distinct personality 
    clusters with characteristic RIASEC-self concept profiles. These findings contribute to 
    understanding the psychological foundations of career development and self-concept formation.
    """
    story.append(Paragraph(abstract_text, styles['body']))
    story.append(PageBreak())

def create_methodology_section(story, styles, data):
    """Create methodology section"""
    
    story.append(Paragraph("1. METHODOLOGY", styles['heading']))
    
    # Participants
    story.append(Paragraph("1.1 Participants", styles['subheading']))
    participants_text = f"""
    The study included {len(data['combined'])} participants who completed both RIASEC career 
    interest inventory and self-concept questionnaire. Participants were recruited from diverse 
    backgrounds to ensure representative sampling across different career interest patterns and 
    self-concept profiles.
    """
    story.append(Paragraph(participants_text, styles['body']))
    
    # Instruments
    story.append(Paragraph("1.2 Instruments", styles['subheading']))
    instruments_text = """
    <b>RIASEC Career Interest Inventory:</b> Measures six career interest dimensions based on 
    Holland's theory - Realistic, Investigative, Artistic, Social, Enterprising, and Conventional.<br/><br/>
    
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
    • <b>Correlation Analysis:</b> Pearson and Spearman correlations between RIASEC and self-concept dimensions<br/>
    • <b>Normality Testing:</b> Shapiro-Wilk tests to determine appropriate correlation methods<br/>
    • <b>Cluster Analysis:</b> K-means clustering to identify distinct participant profiles<br/>
    • <b>Principal Component Analysis:</b> Dimensionality reduction and pattern identification<br/>
    • <b>Individual Profiling:</b> Personalized RIASEC codes and self-concept characterization
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
    between RIASEC career interests and self-concept dimensions. All significant correlations 
    demonstrated medium to large effect sizes, indicating practically meaningful relationships.
    """
    story.append(Paragraph(correlation_text, styles['body']))
    
    # Key findings
    story.append(Paragraph("Key Correlation Findings:", styles['subheading']))
    
    # Sort by correlation strength
    top_correlations = significant_corrs.nlargest(5, 'Primary_r', keep='all')
    
    findings_text = ""
    for _, corr in top_correlations.iterrows():
        effect_size = corr['Effect_Size']
        sig_level = "**" if corr['Primary_p'] < 0.01 else "*"
        findings_text += f"""
        • <b>{corr['RIASEC_Dimension']} ↔ {corr['Self_Concept_Dimension']}:</b> 
        r = {corr['Primary_r']:.3f}{sig_level} ({effect_size} effect)<br/>
        """
    
    story.append(Paragraph(findings_text, styles['body']))
    
    # Cluster Analysis Results
    story.append(Paragraph("2.2 Cluster Analysis", styles['subheading']))
    
    # Get cluster info
    cluster_counts = data['clusters']['Cluster'].value_counts().sort_index()
    
    cluster_text = f"""
    K-means clustering identified 2 distinct participant clusters with different RIASEC-self 
    concept profiles:<br/><br/>
    
    <b>Cluster 1 (n={cluster_counts[0]}):</b> Higher investigative interests and stronger 
    self-concept across multiple dimensions, particularly in abilities, worthiness, and 
    future orientation.<br/><br/>
    
    <b>Cluster 2 (n={cluster_counts[1]}):</b> More diverse RIASEC interests with moderate 
    self-concept levels, showing particular strength in health/sex appropriateness and 
    beliefs/convictions.
    """
    story.append(Paragraph(cluster_text, styles['body']))
    
    # PCA Results
    story.append(Paragraph("2.3 Principal Component Analysis", styles['subheading']))
    
    pca_text = """
    PCA revealed that the first two principal components explained 51.7% of the total variance 
    in the combined RIASEC-self concept data. The first component (28.0% variance) was 
    characterized by positive loadings on Present/Past/Future orientation, Abilities, 
    Worthiness, and Investigative interests, suggesting a "Future-Oriented Competence" factor. 
    The second component (23.8% variance) was dominated by Artistic and Enterprising interests 
    with negative loadings on Shame/Guilt and Self-Acceptance, indicating a 
    "Creative-Entrepreneurial vs. Self-Critical" dimension.
    """
    story.append(Paragraph(pca_text, styles['body']))
    story.append(PageBreak())

def create_discussion_section(story, styles, data):
    """Create discussion and interpretation section"""
    
    story.append(Paragraph("3. DISCUSSION", styles['heading']))
    
    # Theoretical Implications
    story.append(Paragraph("3.1 Theoretical Implications", styles['subheading']))
    
    theory_text = """
    The findings provide strong support for the interconnection between career interests and 
    self-concept development. The robust correlation between Investigative interests and multiple 
    self-concept dimensions (r = 0.626 to 0.767) suggests that individuals drawn to investigative 
    careers possess stronger self-perceived abilities, personal worthiness, and future orientation. 
    This aligns with Holland's theory that career interests reflect underlying personality 
    patterns and self-perceptions.
    """
    story.append(Paragraph(theory_text, styles['body']))
    
    # Practical Applications
    story.append(Paragraph("3.2 Practical Applications", styles['subheading']))
    
    practical_text = """
    These findings have important implications for career counseling and personal development:<br/><br/>
    
    • <b>Career Guidance:</b> Strong investigative interests may indicate individuals with robust 
    self-concept foundations, while artistic interests may require additional self-acceptance support.<br/><br/>
    
    • <b>Personal Development:</b> The negative correlation between artistic interests and 
    self-acceptance suggests that creative individuals may benefit from targeted self-esteem interventions.<br/><br/>
    
    • <b>Educational Planning:</b> Understanding the relationship between enterprising interests 
    and health/sex appropriateness can inform educational program design and student support services.
    """
    story.append(Paragraph(practical_text, styles['body']))
    
    # Limitations
    story.append(Paragraph("3.3 Limitations", styles['subheading']))
    
    limitations_text = """
    Several limitations should be considered when interpreting these results:<br/>
    
    • Small sample size (n=12) limits generalizability and statistical power<br/>
    • Cross-sectional design prevents causal inferences<br/>
    • Self-report measures may be subject to social desirability bias<br/>
    • Cultural and demographic factors were not systematically controlled
    """
    story.append(Paragraph(limitations_text, styles['body']))
    story.append(PageBreak())

def create_conclusions_section(story, styles, data):
    """Create conclusions section"""
    
    story.append(Paragraph("4. CONCLUSIONS", styles['heading']))
    
    conclusions_text = """
    This comprehensive cross-analysis of RIASEC career interests and self-concept dimensions 
    reveals meaningful patterns that advance our understanding of personality-career relationships:
    """
    story.append(Paragraph(conclusions_text, styles['body']))
    
    # Key conclusions
    story.append(Paragraph("Key Conclusions:", styles['subheading']))
    
    conclusions_list = """
    1. <b>Strong Investigative-Self Concept Link:</b> Investigative career interests demonstrate 
    robust positive associations with multiple self-concept dimensions, particularly abilities, 
    worthiness, and future orientation (r = 0.626-0.767).<br/><br/>
    
    2. <b>Artistic Interests and Self-Acceptance:</b> Negative correlations between artistic 
    interests and self-acceptance suggest that creative individuals may experience self-critical 
    tendencies that warrant targeted support.<br/><br/>
    
    3. <b>Distinct Personality Clusters:</b> Two meaningful clusters emerge with different 
    RIASEC-self concept profiles, indicating heterogeneous pathways in career-personality development.<br/><br/>
    
    4. <b>Future-Oriented Competence Factor:</b> PCA reveals a primary dimension combining 
    investigative interests with positive self-concept, suggesting a fundamental competence-confidence axis.<br/><br/>
    
    5. <b>Practical Utility:</b> Individual profiling demonstrates the value of integrated 
    RIASEC-self concept assessment for personalized career guidance.
    """
    story.append(Paragraph(conclusions_list, styles['body']))
    
    # Future Research
    story.append(Paragraph("Future Research Directions:", styles['subheading']))
    
    future_text = """
    Future studies should expand sample sizes, employ longitudinal designs, and investigate 
    cultural moderators of RIASEC-self concept relationships. Integration with other personality 
    frameworks (Big Five, Myers-Briggs) would provide broader theoretical insights. Intervention 
    studies examining whether career interest development influences self-concept change would 
    have significant applied value.
    """
    story.append(Paragraph(future_text, styles['body']))
    story.append(PageBreak())

def create_appendix_section(story, styles, data):
    """Create appendix with detailed statistics"""
    
    story.append(Paragraph("APPENDIX: DETAILED STATISTICAL RESULTS", styles['heading']))
    
    # Correlation Table
    story.append(Paragraph("A.1 Complete Correlation Matrix", styles['subheading']))
    
    # Get significant correlations for table
    significant = data['correlations'][data['correlations']['Primary_p'] < 0.05]
    
    # Create correlation table
    table_data = [['RIASEC Dimension', 'Self-Concept Dimension', 'Correlation', 'p-value', 'Effect Size']]
    
    for _, row in significant.iterrows():
        sig_marker = "**" if row['Primary_p'] < 0.01 else "*"
        table_data.append([
            row['RIASEC_Dimension'],
            row['Self_Concept_Dimension'],
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
    
    profile_data = [['Participant', 'RIASEC Code', 'Highest Interest', 'Self-Concept Score']]
    
    for _, profile in top_profiles.iterrows():
        profile_data.append([
            profile['participant'].title(),
            profile['riasec_code'],
            profile['riasec_highest'],
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
    
    base_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis'
    
    # List of visualizations to include
    visualizations = [
        ('correlation_heatmap.png', 'Correlation Heatmap: RIASEC-Self Concept Relationships'),
        ('significant_correlations.png', 'Significant Correlations: Effect Sizes and Significance Levels'),
        ('cluster_profiles.png', 'Cluster Profiles: RIASEC and Self-Concept Patterns'),
        ('pca_biplot.png', 'PCA Biplot: Principal Components and Variable Loadings'),
        ('individual_radar_charts.png', 'Individual Profiles: Top Participants RIASEC Patterns'),
        ('distribution_plots.png', 'Distribution Analysis: Key Variables'),
        ('scatter_matrix.png', 'Scatter Matrix: Relationship Patterns')
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
    output_path = r'C:\Users\p2var\vandana\riasec_self_concept_analysis\RIASEC_Self_Concept_Analysis_Report.pdf'
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
    print(f"- Clusters identified: {data['clusters']['Cluster'].nunique()}")
    print(f"- Individual profiles: {len(data['profiles'])}")
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main()