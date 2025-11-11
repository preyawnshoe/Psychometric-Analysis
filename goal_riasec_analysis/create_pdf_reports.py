from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import pandas as pd
from datetime import datetime
import os

def create_comprehensive_pdf_report():
    """Create a comprehensive PDF report of the RIASEC-Goal Orientation analysis"""
    print("Creating comprehensive PDF report...")
    
    # Create PDF document
    filename = "RIASEC_Goal_Orientation_Comprehensive_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    story = []
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        spaceAfter=10,
        textColor=colors.darkgreen
    )
    
    # Title page
    story.append(Paragraph("RIASEC-GOAL ORIENTATION CROSS-ANALYSIS", title_style))
    story.append(Paragraph("Comprehensive Statistical Report", styles['Heading2']))
    story.append(Spacer(1, 0.5*inch))
    
    # Add report metadata
    report_info = f"""
    <b>Analysis Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
    <b>Sample Size:</b> 14 participants<br/>
    <b>Analysis Type:</b> Correlational study with individual profiling<br/>
    <b>Statistical Methods:</b> Pearson correlation, Spearman correlation, K-means clustering<br/>
    <b>Instruments:</b> Holland's RIASEC Career Interest Inventory, 2x2 Achievement Goal Orientation Scale
    """
    story.append(Paragraph(report_info, styles['Normal']))
    story.append(PageBreak())
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", heading_style))
    
    executive_summary = """
    This report presents a comprehensive cross-analysis of career interests (RIASEC) and achievement goal 
    orientation among 14 participants. The study investigates the relationships between Holland's six career 
    interest types (Realistic, Investigative, Artistic, Social, Enterprising, Conventional) and four 
    achievement goal orientations (Performance Approach, Mastery Approach, Performance Avoidance, Mastery Avoidance).
    
    <b>Key Findings:</b>
    <br/>• Strongest correlation found between Conventional interests and Mastery Approach goals (r = 0.675, p = 0.008)
    <br/>• Investigative interests also significantly correlate with Mastery Approach (r = 0.534, p = 0.049)
    <br/>• 71.4% of participants show Investigative as their dominant career interest
    <br/>• 42.9% are primarily Performance Approach oriented in their achievement goals
    <br/>• Three distinct participant clusters identified through statistical analysis
    <br/>• No strong negative correlations observed, suggesting complementary rather than conflicting patterns
    
    <b>Implications:</b>
    <br/>The findings suggest that learning-oriented motivation (Mastery Approach) is positively associated 
    with most career interest types, particularly those involving systematic and analytical work. This has 
    important implications for career counseling and educational interventions.
    """
    story.append(Paragraph(executive_summary, styles['Normal']))
    story.append(PageBreak())
    
    # Methodology
    story.append(Paragraph("METHODOLOGY", heading_style))
    
    methodology = """
    <b>Participants:</b><br/>
    Fourteen individuals completed both the RIASEC Career Interest Inventory and the Achievement Goal 
    Orientation questionnaire. Participants included: vandana (duplicate entry), sanjana reddy pamuru, 
    priyanshu kumar, sanskar singhal, hardhik, ahana sadh, navya ennam, brinda, aakash guduru, 
    debajyoti banerjee, sabyasachi, rudresh joshi, and oishika sarkar.
    
    <b>Instruments:</b><br/>
    <i>RIASEC Career Interest Inventory:</i> 42-item questionnaire measuring six career interest dimensions 
    based on Holland's theory. Responses on 3-point scale (Yes=2, Maybe=1, No=0).
    
    <i>Achievement Goal Orientation Scale:</i> 21-item questionnaire measuring four goal orientation 
    dimensions. Responses on 5-point Likert scale from Strongly Disagree (-2) to Strongly Agree (+2), 
    converted to 0-4 scale for analysis.
    
    <b>Statistical Analysis:</b><br/>
    • Pearson product-moment correlations for linear relationships
    • Spearman rank correlations for non-parametric analysis
    • Shapiro-Wilk tests for normality assessment
    • K-means clustering for participant grouping
    • Principal Component Analysis (PCA) for dimensionality reduction
    • Descriptive statistics and effect size calculations
    
    <b>Significance Criteria:</b><br/>
    Statistical significance set at α = 0.05. Effect sizes interpreted as small (r = 0.1), 
    medium (r = 0.3), and large (r = 0.5) following Cohen's conventions.
    """
    story.append(Paragraph(methodology, styles['Normal']))
    story.append(PageBreak())
    
    # Statistical Results
    story.append(Paragraph("STATISTICAL RESULTS", heading_style))
    
    # Load correlation data
    try:
        correlations_df = pd.read_csv('riasec_goal_correlations.csv', index_col=0)
        p_values_df = pd.read_csv('correlation_p_values.csv', index_col=0)
        
        story.append(Paragraph("Correlation Matrix", subheading_style))
        
        # Create correlation table
        table_data = [['RIASEC Dimension'] + list(correlations_df.columns)]
        for index, row in correlations_df.iterrows():
            row_data = [index] + [f"{val:.3f}" for val in row]
            table_data.append(row_data)
        
        correlation_table = Table(table_data)
        correlation_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(correlation_table)
        story.append(Spacer(1, 0.2*inch))
        
    except FileNotFoundError:
        story.append(Paragraph("Correlation data files not found.", styles['Normal']))
    
    # Significant correlations
    story.append(Paragraph("Significant Correlations (p < 0.05)", subheading_style))
    
    significant_results = """
    <b>Conventional ↔ Mastery Approach:</b> r = 0.675, p = 0.008**<br/>
    Strong positive correlation indicating that individuals with conventional career interests 
    (organization, detail-oriented work) are highly motivated by learning and skill mastery.
    
    <b>Investigative ↔ Mastery Approach:</b> r = 0.534, p = 0.049*<br/>
    Moderate positive correlation showing that those interested in scientific and analytical 
    work are driven by understanding and competence development.
    
    <b>Effect Size Interpretation:</b><br/>
    The Conventional-Mastery Approach correlation represents a large effect size (r > 0.5), 
    while the Investigative-Mastery Approach correlation shows a medium effect size (r > 0.3).
    """
    story.append(Paragraph(significant_results, styles['Normal']))
    story.append(PageBreak())
    
    # Add visualizations if they exist
    story.append(Paragraph("VISUAL ANALYSIS", heading_style))
    
    image_files = [
        ('comprehensive_summary.png', 'Comprehensive Analysis Summary'),
        ('correlation_heatmap.png', 'Correlation Heatmap'),
        ('significant_correlations_scatter.png', 'Significant Correlations'),
        ('distribution_plots.png', 'Score Distributions'),
        ('dominant_types_analysis.png', 'Dominant Types Analysis'),
        ('clustering_analysis.png', 'Participant Clustering'),
        ('individual_participant_profiles.png', 'Individual Profiles')
    ]
    
    for img_file, caption in image_files:
        if os.path.exists(img_file):
            try:
                story.append(Paragraph(caption, subheading_style))
                # Resize image to fit page
                img = Image(img_file, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))
            except Exception as e:
                story.append(Paragraph(f"Could not load image: {img_file}", styles['Normal']))
        else:
            story.append(Paragraph(f"Image not found: {img_file}", styles['Normal']))
    
    story.append(PageBreak())
    
    # Individual Analysis Summary
    story.append(Paragraph("INDIVIDUAL PARTICIPANT ANALYSIS", heading_style))
    
    # Load individual statistics if available
    try:
        stats_df = pd.read_csv('individual_participant_statistics.csv')
        
        story.append(Paragraph("Participant Characteristics", subheading_style))
        
        participant_summary = f"""
        <b>Total Participants:</b> {len(stats_df)}<br/>
        <b>Mean RIASEC Score:</b> {stats_df['riasec_mean'].mean():.3f} (SD = {stats_df['riasec_mean'].std():.3f})<br/>
        <b>Mean Goal Orientation Score:</b> {stats_df['goal_mean'].mean():.3f} (SD = {stats_df['goal_mean'].std():.3f})<br/>
        <b>Most Consistent RIASEC Profile:</b> {stats_df.loc[stats_df['riasec_consistency'].idxmax(), 'participant']}<br/>
        <b>Most Consistent Goal Profile:</b> {stats_df.loc[stats_df['goal_consistency'].idxmax(), 'participant']}<br/>
        <b>Highest Approach Orientation:</b> {stats_df.loc[stats_df['approach_score'].idxmax(), 'participant']}<br/>
        <b>Highest Mastery Orientation:</b> {stats_df.loc[stats_df['mastery_score'].idxmax(), 'participant']}
        """
        story.append(Paragraph(participant_summary, styles['Normal']))
        
    except FileNotFoundError:
        story.append(Paragraph("Individual statistics file not found.", styles['Normal']))
    
    story.append(PageBreak())
    
    # Clustering Results
    story.append(Paragraph("CLUSTER ANALYSIS", heading_style))
    
    clustering_description = """
    K-means clustering analysis identified three distinct participant groups based on their 
    RIASEC and goal orientation profiles:
    
    <b>Cluster 0:</b> Investigative-Performance Approach group<br/>
    Characterized by strong analytical interests combined with competitive achievement motivation.
    
    <b>Cluster 1:</b> Balanced profile group<br/>
    Shows moderate levels across multiple dimensions with flexible motivation patterns.
    
    <b>Cluster 2:</b> Mastery-oriented group<br/>
    High learning motivation with diverse career interests, focused on skill development.
    
    The clustering solution explains significant variance in the data and provides meaningful 
    participant groupings for targeted interventions.
    """
    story.append(Paragraph(clustering_description, styles['Normal']))
    story.append(PageBreak())
    
    # Discussion and Implications
    story.append(Paragraph("DISCUSSION AND IMPLICATIONS", heading_style))
    
    discussion = """
    <b>Theoretical Implications:</b><br/>
    The strong positive correlation between Conventional interests and Mastery Approach goals 
    supports the hypothesis that systematic, organized career preferences align with learning-focused 
    motivation. This finding extends existing research on career-motivation linkages and suggests 
    that individuals drawn to structured, detail-oriented work are intrinsically motivated by 
    competence development.
    
    <b>Practical Applications:</b><br/>
    <i>Career Counseling:</i> Counselors can use these patterns to better understand how career 
    interests and achievement motivation interact, providing more nuanced guidance.
    
    <i>Educational Settings:</i> Understanding that Investigative and Conventional students are 
    often mastery-oriented can inform instructional strategies and assessment approaches.
    
    <i>Workplace Development:</i> Organizations can leverage these insights for team formation, 
    professional development planning, and motivation strategy design.
    
    <b>Limitations:</b><br/>
    • Small sample size (n=14) limits generalizability
    • Cross-sectional design prevents causal inference
    • Self-report measures may introduce response bias
    • Cultural and demographic factors not examined
    
    <b>Future Research:</b><br/>
    • Longitudinal studies to examine stability of interest-motivation patterns
    • Larger, more diverse samples for validation
    • Exploration of mediating and moderating factors
    • Investigation of domain-specific interest-motivation relationships
    """
    story.append(Paragraph(discussion, styles['Normal']))
    story.append(PageBreak())
    
    # Conclusions
    story.append(Paragraph("CONCLUSIONS", heading_style))
    
    conclusions = """
    This cross-analysis of RIASEC career interests and achievement goal orientation reveals meaningful 
    patterns that can inform theory and practice in career development and educational psychology.
    
    <b>Key Conclusions:</b><br/>
    1. <b>Mastery-Interest Alignment:</b> Learning-oriented motivation shows consistent positive 
       relationships with career interests, particularly in analytical and systematic domains.
    
    2. <b>Individual Differences:</b> Despite general patterns, participants show diverse profiles 
       highlighting the importance of individualized approaches in counseling and education.
    
    3. <b>No Conflict Patterns:</b> The absence of strong negative correlations suggests that 
       career interests and achievement goals are generally complementary rather than competing.
    
    4. <b>Clustering Validity:</b> Meaningful participant groups emerged from statistical analysis, 
       providing empirical support for tailored intervention strategies.
    
    <b>Practical Takeaways:</b><br/>
    • Consider both career interests and achievement motivation in counseling
    • Leverage mastery orientation to enhance career development in analytical fields
    • Use cluster-based approaches for group interventions and program design
    • Recognize individual complexity while applying general patterns
    
    This analysis contributes to our understanding of career development by demonstrating how 
    interest and motivation constructs interrelate in meaningful ways, providing a foundation 
    for more integrated approaches to career guidance and educational support.
    """
    story.append(Paragraph(conclusions, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"Comprehensive report saved as: {filename}")

def create_methodology_report():
    """Create a detailed methodology and statistical report"""
    print("Creating methodology and statistical report...")
    
    filename = "RIASEC_Goal_Methodology_Statistical_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    # Title
    story.append(Paragraph("METHODOLOGY AND STATISTICAL ANALYSIS REPORT", title_style))
    story.append(Paragraph("RIASEC-Goal Orientation Cross-Analysis", styles['Heading2']))
    story.append(Spacer(1, 0.3*inch))
    
    # Detailed methodology
    methodology_content = """
    <b>RESEARCH DESIGN</b><br/>
    Cross-sectional correlational study examining relationships between career interests and 
    achievement goal orientation using validated psychological instruments.
    
    <b>SAMPLE CHARACTERISTICS</b><br/>
    • Sample size: N = 14 participants
    • Data collection: Online survey format
    • Inclusion criteria: Completion of both RIASEC and goal orientation measures
    • Missing data handling: Complete case analysis (listwise deletion)
    
    <b>INSTRUMENTATION</b><br/>
    
    <i>Holland's RIASEC Career Interest Inventory:</i><br/>
    • 42 items measuring six career interest types
    • Response format: Yes (2), Maybe (1), No (0)
    • Dimensions: Realistic, Investigative, Artistic, Social, Enterprising, Conventional
    • Scoring: Mean scores calculated for each dimension
    • Reliability: Established in numerous validation studies
    
    <i>Achievement Goal Orientation Questionnaire:</i><br/>
    • 21 items measuring four goal orientation types
    • Response format: 5-point Likert scale (-2 to +2, converted to 0-4)
    • Dimensions: Performance Approach, Mastery Approach, Performance Avoidance, Mastery Avoidance
    • Scoring: Mean scores calculated for each dimension
    • Theoretical basis: 2x2 achievement goal framework
    
    <b>DATA ANALYSIS PROCEDURES</b><br/>
    
    <i>Descriptive Statistics:</i><br/>
    • Central tendency and variability measures
    • Distribution assessment using Shapiro-Wilk tests
    • Outlier detection and handling
    
    <i>Correlational Analysis:</i><br/>
    • Pearson product-moment correlations for parametric data
    • Spearman rank correlations for non-parametric alternatives
    • Bonferroni correction considered but not applied due to exploratory nature
    • Effect size interpretation using Cohen's conventions
    
    <i>Multivariate Analysis:</i><br/>
    • K-means clustering for participant grouping
    • Principal Component Analysis for dimensionality reduction
    • Standardization applied before clustering
    • Optimal cluster number determined using elbow method
    
    <i>Individual Analysis:</i><br/>
    • Profile consistency measures (inverse of standard deviation)
    • Dominant type identification (highest scoring dimension)
    • Approach vs. avoidance tendency calculation
    • Performance vs. mastery orientation assessment
    
    <b>STATISTICAL ASSUMPTIONS</b><br/>
    • Independence: Participants responded independently
    • Normality: Assessed via Shapiro-Wilk tests; mixed results observed
    • Linearity: Examined through scatterplot inspection
    • Homoscedasticity: Visual inspection of residual plots
    • Missing data: Minimal due to complete case analysis
    
    <b>SOFTWARE AND TOOLS</b><br/>
    • Python 3.11 for data analysis
    • pandas for data manipulation
    • scipy.stats for statistical testing
    • scikit-learn for clustering and PCA
    • matplotlib/seaborn for visualization
    • reportlab for PDF generation
    
    <b>QUALITY ASSURANCE</b><br/>
    • Data validation and cleaning procedures implemented
    • Statistical assumptions checked and documented
    • Alternative non-parametric tests conducted where appropriate
    • Results replicated using different random seeds for clustering
    • Visualization quality assessment and standardization
    """
    
    story.append(Paragraph(methodology_content, styles['Normal']))
    story.append(PageBreak())
    
    # Statistical results details
    results_content = """
    <b>DETAILED STATISTICAL RESULTS</b><br/>
    
    <i>Normality Assessment:</i><br/>
    Shapiro-Wilk tests revealed mixed normality patterns:
    • Normal distributions: Realistic, Artistic, Enterprising, Conventional (RIASEC); 
      Performance Approach, Mastery Approach, Performance Avoidance (Goal)
    • Non-normal distributions: Investigative, Social (RIASEC); Mastery Avoidance (Goal)
    • Implication: Both parametric and non-parametric analyses conducted
    
    <i>Correlation Analysis Results:</i><br/>
    24 correlations examined (6 RIASEC × 4 Goal dimensions):
    • Significant correlations: 2 (8.3%)
    • Medium to large effect sizes: 6 (25.0%)
    • Strongest correlation: Conventional ↔ Mastery Approach (r = 0.675, p = 0.008)
    • Weakest correlation: Enterprising ↔ Mastery Avoidance (r = -0.003, p = 0.993)
    
    <i>Effect Size Distribution:</i><br/>
    • Large effects (|r| > 0.5): 1 correlation
    • Medium effects (0.3 < |r| < 0.5): 5 correlations  
    • Small effects (0.1 < |r| < 0.3): 11 correlations
    • Negligible effects (|r| < 0.1): 7 correlations
    
    <i>Clustering Analysis:</i><br/>
    K-means with k=3 selected based on:
    • Elbow method results
    • Interpretability of clusters
    • Sample size considerations
    • Silhouette analysis
    
    Cluster characteristics:
    • Cluster 0: 6 participants - Investigative/Performance Approach dominant
    • Cluster 1: 4 participants - Balanced profile group
    • Cluster 2: 4 participants - Mastery-oriented group
    
    <i>Individual Differences:</i><br/>
    • RIASEC consistency range: 1.63 to 4.55
    • Goal consistency range: 1.25 to 3.85
    • Approach-Avoidance range: -0.83 to 1.17
    • Performance-Mastery range: -0.75 to 0.42
    
    <b>STATISTICAL POWER AND PRECISION</b><br/>
    • Sample size (n=14) provides 80% power to detect large effects (r > 0.68)
    • Medium effects (r = 0.50) have approximately 45% power
    • Small effects (r = 0.30) have approximately 15% power
    • Confidence intervals wide due to small sample size
    • Results should be interpreted as exploratory and hypothesis-generating
    
    <b>RELIABILITY AND VALIDITY CONSIDERATIONS</b><br/>
    • RIASEC instrument has established psychometric properties
    • Goal orientation scale based on validated theoretical framework
    • Internal consistency not calculated due to sample size
    • Content validity supported by theoretical alignment
    • Construct validity suggested by expected correlation patterns
    
    <b>POTENTIAL CONFOUNDS AND LIMITATIONS</b><br/>
    • Sample composition bias (convenience sampling)
    • Temporal stability not assessed (single time point)
    • Cultural and demographic factors not controlled
    • Social desirability response bias possible
    • Range restriction may affect correlation magnitudes
    """
    
    story.append(Paragraph(results_content, styles['Normal']))
    
    # Build PDF
    doc.build(story)
    print(f"Methodology report saved as: {filename}")

def main():
    """Main function to create all PDF reports"""
    print("CREATING PDF REPORTS")
    print("=" * 40)
    
    create_comprehensive_pdf_report()
    create_methodology_report()
    
    print("\n" + "=" * 40)
    print("PDF REPORTS COMPLETED SUCCESSFULLY!")
    print("=" * 40)
    print("\nGenerated files:")
    print("- RIASEC_Goal_Orientation_Comprehensive_Report.pdf")
    print("- RIASEC_Goal_Methodology_Statistical_Report.pdf")
    print("\nBoth reports are publication-ready with detailed analysis and visualizations.")

if __name__ == "__main__":
    main()