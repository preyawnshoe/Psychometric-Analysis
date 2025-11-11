#!/usr/bin/env python3
"""
RIASEC PDF Report Generator
Creates a comprehensive PDF report combining statistical analysis and visualizations.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from datetime import datetime
import pandas as pd
import os

def create_custom_styles():
    """Create custom paragraph styles for the report."""
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    # Section header style
    section_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    # Subsection style
    subsection_style = ParagraphStyle(
        'SubsectionHeader',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=15,
        textColor=colors.navy,
        fontName='Helvetica-Bold'
    )
    
    # Body text style
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Bullet point style
    bullet_style = ParagraphStyle(
        'BulletPoint',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=4,
        leftIndent=20,
        fontName='Helvetica'
    )
    
    return {
        'title': title_style,
        'section': section_style,
        'subsection': subsection_style,
        'body': body_style,
        'bullet': bullet_style,
        'normal': styles['Normal']
    }

def load_data_for_summary():
    """Load analysis data for summary statistics."""
    try:
        scores_df = pd.read_csv('riasec_individual_scores.csv', index_col=0)
        desc_stats = pd.read_csv('riasec_descriptive_stats.csv', index_col=0)
        return scores_df, desc_stats
    except Exception as e:
        print(f"Warning: Could not load data files: {e}")
        return None, None

def create_cover_page(story, styles):
    """Create the cover page of the report."""
    # Title
    title = Paragraph("RIASEC Interest Inventory", styles['title'])
    story.append(title)
    story.append(Spacer(1, 0.5*inch))
    
    # Subtitle
    subtitle = Paragraph("Comprehensive Statistical Analysis Report", styles['section'])
    story.append(subtitle)
    story.append(Spacer(1, 1*inch))
    
    # Report details
    report_info = f"""
    <para align="center">
    <b>Analysis Date:</b> {datetime.now().strftime("%B %d, %Y")}<br/>
    <b>Sample Size:</b> 16 Participants<br/>
    <b>Assessment Type:</b> Holland's Career Interest Inventory<br/>
    <b>Report Type:</b> Descriptive Statistical Analysis with Visualizations
    </para>
    """
    story.append(Paragraph(report_info, styles['body']))
    story.append(Spacer(1, 1*inch))
    
    # RIASEC explanation
    riasec_explanation = """
    <para align="center">
    <b>RIASEC Categories:</b><br/>
    <b>R</b> - Realistic (Hands-on, practical activities)<br/>
    <b>I</b> - Investigative (Analytical, scientific thinking)<br/>
    <b>A</b> - Artistic (Creative, expressive activities)<br/>
    <b>S</b> - Social (Helping, teaching others)<br/>
    <b>E</b> - Enterprising (Leading, persuading others)<br/>
    <b>C</b> - Conventional (Organizing, detail-oriented tasks)
    </para>
    """
    story.append(Paragraph(riasec_explanation, styles['body']))
    
    story.append(PageBreak())

def create_executive_summary(story, styles, scores_df, desc_stats):
    """Create executive summary section."""
    story.append(Paragraph("Executive Summary", styles['section']))
    
    if scores_df is not None and desc_stats is not None:
        # Calculate key statistics
        category_means = desc_stats.loc['mean'].sort_values(ascending=False)
        dominant_types = scores_df.idxmax(axis=1)
        type_counts = dominant_types.value_counts()
        overall_mean = scores_df.values.mean()
        
        # Correlations
        correlations = scores_df.corr()
        strongest_corr = 0
        strongest_pair = ""
        for i in range(len(correlations.columns)):
            for j in range(i+1, len(correlations.columns)):
                corr_val = abs(correlations.iloc[i, j])
                if corr_val > strongest_corr:
                    strongest_corr = correlations.iloc[i, j]
                    strongest_pair = f"{correlations.columns[i]}-{correlations.columns[j]}"
        
        summary_text = f"""
        This comprehensive report analyzes RIASEC career interest data from 16 participants, 
        providing insights into career preferences and patterns within the group.
        
        <b>Key Findings:</b>
        • <b>Highest Interest Category:</b> {category_means.index[0]} ({category_means.iloc[0]:.3f} mean score)
        • <b>Most Common Dominant Type:</b> {type_counts.index[0]} ({type_counts.iloc[0]} participants, {(type_counts.iloc[0]/len(scores_df)*100):.1f}%)
        • <b>Group Average Score:</b> {overall_mean:.3f} on 1-3 scale
        • <b>Strongest Correlation:</b> {strongest_pair} (r = {strongest_corr:.3f})
        
        <b>Analysis Components:</b>
        • Descriptive statistics for all RIASEC categories
        • Individual participant profiles and Holland codes
        • Correlation analysis between interest areas
        • Visual representations of data patterns
        • Career development recommendations
        """
    else:
        summary_text = """
        This comprehensive report analyzes RIASEC career interest data, providing insights 
        into career preferences and patterns within the assessed group.
        
        The analysis includes descriptive statistics, individual profiles, correlation 
        analysis, and professional visualizations to support career counseling and 
        development programs.
        """
    
    story.append(Paragraph(summary_text, styles['body']))
    story.append(PageBreak())

def add_image_with_caption(story, styles, image_path, caption, max_width=6*inch):
    """Add an image with caption to the story."""
    if os.path.exists(image_path):
        try:
            # Calculate appropriate size
            img = Image(image_path)
            img_width, img_height = img.drawWidth, img.drawHeight
            
            # Scale to fit page if necessary
            if img_width > max_width:
                scale_factor = max_width / img_width
                img.drawWidth = max_width
                img.drawHeight = img_height * scale_factor
            
            story.append(img)
            story.append(Spacer(1, 0.1*inch))
            story.append(Paragraph(f"<i>{caption}</i>", styles['body']))
            story.append(Spacer(1, 0.3*inch))
            return True
        except Exception as e:
            print(f"Error adding image {image_path}: {e}")
            story.append(Paragraph(f"<i>Image not available: {caption}</i>", styles['body']))
            return False
    else:
        story.append(Paragraph(f"<i>Image not found: {caption}</i>", styles['body']))
        return False

def create_visualizations_section(story, styles):
    """Create the visualizations section."""
    story.append(Paragraph("Statistical Visualizations", styles['section']))
    
    # Distribution Analysis
    story.append(Paragraph("Distribution Analysis", styles['subsection']))
    story.append(Paragraph(
        "The following charts show the distribution patterns of RIASEC scores across all participants, "
        "including measures of central tendency, variability, and distribution shapes.",
        styles['body']
    ))
    
    add_image_with_caption(
        story, styles,
        'riasec_distribution_analysis.png',
        "Figure 1: RIASEC Score Distributions - Box plots, means with error bars, dominant type distribution, and violin plots"
    )
    
    story.append(PageBreak())
    
    # Correlation and Profiles
    story.append(Paragraph("Correlation Analysis and Individual Profiles", styles['subsection']))
    story.append(Paragraph(
        "These visualizations examine relationships between RIASEC categories and show individual "
        "participant patterns across the six interest areas.",
        styles['body']
    ))
    
    add_image_with_caption(
        story, styles,
        'riasec_correlation_profiles.png',
        "Figure 2: Correlation matrix, individual profiles, score comparisons, and summary statistics"
    )
    
    story.append(PageBreak())
    
    # Individual Analysis
    story.append(Paragraph("Individual Participant Analysis", styles['subsection']))
    story.append(Paragraph(
        "Each participant's RIASEC profile is displayed separately, with dominant types highlighted "
        "and exact scores labeled for detailed individual analysis.",
        styles['body']
    ))
    
    add_image_with_caption(
        story, styles,
        'riasec_individual_analysis.png',
        "Figure 3: Individual RIASEC profiles for all participants with dominant types highlighted"
    )

def create_statistics_summary_table(story, styles, desc_stats):
    """Create a summary statistics table."""
    if desc_stats is None:
        story.append(Paragraph("Statistical data not available for table generation.", styles['body']))
        return
    
    story.append(PageBreak())
    story.append(Paragraph("Statistical Summary Table", styles['section']))
    
    # Prepare table data
    table_data = [['Statistic', 'R', 'I', 'A', 'S', 'E', 'C']]
    
    for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        row = [stat.title()]
        for category in ['R', 'I', 'A', 'S', 'E', 'C']:
            row.append(f"{desc_stats.loc[stat, category]:.3f}")
        table_data.append(row)
    
    # Create table
    table = Table(table_data, colWidths=[1.2*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
    
    # Style the table
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.3*inch))
    
    # Table explanation
    explanation = """
    The table above presents key descriptive statistics for each RIASEC category:
    • <b>Mean:</b> Average score across all participants
    • <b>Std:</b> Standard deviation (measure of variability)
    • <b>Min/Max:</b> Lowest and highest individual scores
    • <b>25%, 50%, 75%:</b> Quartile values (percentiles)
    """
    story.append(Paragraph(explanation, styles['body']))

def create_individual_profiles_section(story, styles, scores_df):
    """Create individual profiles section."""
    if scores_df is None:
        return
    
    story.append(PageBreak())
    story.append(Paragraph("Individual Participant Profiles", styles['section']))
    
    story.append(Paragraph(
        "Each participant's Holland Code and dominant interests are summarized below. "
        "The Holland Code represents the top three interest areas in order of preference.",
        styles['body']
    ))
    story.append(Spacer(1, 0.2*inch))
    
    # Create profile data for each participant
    for participant in scores_df.index:
        scores = scores_df.loc[participant]
        dominant_type = scores.idxmax()
        dominant_score = scores.max()
        holland_code = ''.join(scores.nlargest(3).index)
        mean_score = scores.mean()
        
        profile_text = f"""
        <b>{participant}:</b><br/>
        • Holland Code: {holland_code}<br/>
        • Dominant Interest: {dominant_type} (Score: {dominant_score:.3f})<br/>
        • Average Score: {mean_score:.3f}<br/>
        """
        
        story.append(Paragraph(profile_text, styles['body']))
        story.append(Spacer(1, 0.1*inch))

def create_recommendations_section(story, styles):
    """Create career recommendations section."""
    story.append(PageBreak())
    story.append(Paragraph("Career Development Recommendations", styles['section']))
    
    recommendations_text = """
    Based on the RIASEC analysis results, the following recommendations are provided for 
    career development and counseling:
    
    <b>Group-Level Recommendations:</b>
    
    <b>1. Investigative Focus:</b> Given the high prevalence of Investigative interests, 
    consider developing programs that emphasize research, analysis, critical thinking, 
    and scientific reasoning skills.
    
    <b>2. Practical Application:</b> The strong Realistic interests suggest value in 
    hands-on learning experiences and practical application of knowledge.
    
    <b>3. Individual Counseling:</b> The diversity of profiles indicates the need for 
    individualized career counseling rather than one-size-fits-all approaches.
    
    <b>4. Hybrid Career Paths:</b> Strong correlations between certain categories suggest 
    exploring careers that combine multiple interest areas (e.g., research with practical 
    application, creative problem-solving).
    
    <b>Individual-Level Recommendations:</b>
    
    <b>For Highly Differentiated Profiles:</b> Focus on specialized career paths that 
    align closely with the dominant interest type. These individuals may thrive in 
    careers with clear specialization.
    
    <b>For Flat Profiles:</b> Explore careers requiring diverse skills and interests. 
    These individuals may excel in interdisciplinary roles or leadership positions 
    requiring broad competencies.
    
    <b>For Moderate Profiles:</b> Consider careers that combine the top 2-3 interest 
    areas, allowing for both specialization and variety.
    
    <b>Next Steps:</b>
    • Conduct follow-up assessments to track interest development
    • Provide career exploration activities aligned with identified interests
    • Connect participants with professionals in their areas of interest
    • Develop action plans based on individual Holland codes
    """
    
    story.append(Paragraph(recommendations_text, styles['body']))

def create_methodology_section(story, styles):
    """Create methodology and technical details section."""
    story.append(PageBreak())
    story.append(Paragraph("Methodology and Technical Details", styles['section']))
    
    methodology_text = """
    <b>Assessment Instrument:</b>
    This analysis is based on a 42-item RIASEC interest inventory following Holland's 
    theoretical framework. Questions were carefully mapped to the six RIASEC categories 
    based on established career interest research.
    
    <b>Scoring Method:</b>
    • Response Scale: Yes (3), Maybe (2), No (1)
    • Category Scores: Average of relevant item responses
    • Holland Code: Top three categories in order of score
    
    <b>Statistical Analysis:</b>
    • Descriptive statistics calculated using standard methods
    • Correlation analysis using Pearson product-moment correlation
    • Normality testing performed using Shapiro-Wilk test
    • Outlier detection using Interquartile Range (IQR) method
    
    <b>Sample Characteristics:</b>
    • Sample Size: 16 participants
    • Response Rate: 100% (no missing data)
    • Assessment Period: September 2025
    
    <b>Limitations:</b>
    • Cross-sectional data (single time point)
    • Self-reported interests (subject to social desirability bias)
    • Sample size limitations for statistical generalization
    • Cultural context considerations
    
    <b>Reliability and Validity:</b>
    • Internal consistency supported by reasonable correlation patterns
    • Face validity confirmed through established RIASEC item mapping
    • Construct validity evidenced by expected category relationships
    """
    
    story.append(Paragraph(methodology_text, styles['body']))

def create_appendix(story, styles):
    """Create appendix with additional information."""
    story.append(PageBreak())
    story.append(Paragraph("Appendix: RIASEC Category Descriptions", styles['section']))
    
    riasec_descriptions = {
        'R - Realistic': """
        <b>Characteristics:</b> Practical, hands-on problem solvers who prefer working with 
        tools, machines, and physical materials. Value tangible results and concrete outcomes.
        
        <b>Typical Activities:</b> Building, repairing, operating machinery, working outdoors, 
        physical coordination tasks.
        
        <b>Career Examples:</b> Engineer, Mechanic, Carpenter, Veterinarian, Farmer, Pilot, 
        Electrician, Chef.
        """,
        
        'I - Investigative': """
        <b>Characteristics:</b> Analytical thinkers who enjoy research, problem-solving, and 
        working with ideas and theories. Value intellectual challenges and scientific reasoning.
        
        <b>Typical Activities:</b> Researching, analyzing data, conducting experiments, 
        theoretical problem-solving.
        
        <b>Career Examples:</b> Scientist, Researcher, Physician, Mathematician, Psychologist, 
        Data Analyst, Professor.
        """,
        
        'A - Artistic': """
        <b>Characteristics:</b> Creative individuals who value self-expression, aesthetics, 
        and working in unstructured environments. Appreciate beauty and originality.
        
        <b>Typical Activities:</b> Creating art, writing, performing, designing, innovating, 
        expressing ideas creatively.
        
        <b>Career Examples:</b> Artist, Designer, Writer, Musician, Actor, Photographer, 
        Graphic Designer.
        """,
        
        'S - Social': """
        <b>Characteristics:</b> People-oriented individuals who enjoy helping, teaching, and 
        working with others in supportive roles. Value interpersonal relationships.
        
        <b>Typical Activities:</b> Teaching, counseling, helping others, working in teams, 
        providing services.
        
        <b>Career Examples:</b> Teacher, Counselor, Social Worker, Nurse, Therapist, Coach, 
        Minister.
        """,
        
        'E - Enterprising': """
        <b>Characteristics:</b> Ambitious leaders who enjoy persuading, managing, and taking 
        on business challenges. Value achievement and influence.
        
        <b>Typical Activities:</b> Leading teams, selling, managing projects, negotiating, 
        starting businesses.
        
        <b>Career Examples:</b> Manager, Entrepreneur, Lawyer, Sales Manager, Marketing Director, 
        CEO, Politician.
        """,
        
        'C - Conventional': """
        <b>Characteristics:</b> Detail-oriented individuals who prefer structured environments 
        and systematic approaches to work. Value order and accuracy.
        
        <b>Typical Activities:</b> Organizing information, following procedures, record keeping, 
        data management.
        
        <b>Career Examples:</b> Accountant, Secretary, Banker, Administrator, Bookkeeper, 
        Data Entry Clerk.
        """
    }
    
    for category, description in riasec_descriptions.items():
        story.append(Paragraph(category, styles['subsection']))
        story.append(Paragraph(description, styles['body']))
        story.append(Spacer(1, 0.2*inch))

def main():
    """Main function to create the PDF report."""
    print("RIASEC PDF Report Generator")
    print("=" * 40)
    
    # Check for required image files
    required_images = [
        'riasec_distribution_analysis.png',
        'riasec_correlation_profiles.png',
        'riasec_individual_analysis.png'
    ]
    
    missing_images = [img for img in required_images if not os.path.exists(img)]
    if missing_images:
        print(f"Warning: Missing image files: {missing_images}")
        print("Some visualizations may not appear in the PDF.")
    
    try:
        # Load data
        scores_df, desc_stats = load_data_for_summary()
        
        # Create PDF document
        filename = f"RIASEC_Comprehensive_Report_{datetime.now().strftime('%Y%m%d')}.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter, topMargin=1*inch, bottomMargin=1*inch)
        
        # Create custom styles
        styles = create_custom_styles()
        
        # Build story (content)
        story = []
        
        # Add all sections
        create_cover_page(story, styles)
        create_executive_summary(story, styles, scores_df, desc_stats)
        create_visualizations_section(story, styles)
        create_statistics_summary_table(story, styles, desc_stats)
        create_individual_profiles_section(story, styles, scores_df)
        create_recommendations_section(story, styles)
        create_methodology_section(story, styles)
        create_appendix(story, styles)
        
        # Build PDF
        print("Generating PDF report...")
        doc.build(story)
        
        print("\n" + "="*50)
        print("PDF REPORT GENERATED SUCCESSFULLY!")
        print("="*50)
        print(f"Report saved as: {filename}")
        print(f"File size: {os.path.getsize(filename) / 1024:.1f} KB")
        print("\nReport includes:")
        print("• Cover page with assessment details")
        print("• Executive summary with key findings")
        print("• All statistical visualizations")
        print("• Summary statistics table")
        print("• Individual participant profiles")
        print("• Career development recommendations")
        print("• Methodology and technical details")
        print("• RIASEC category descriptions")
        
    except Exception as e:
        print(f"Error generating PDF report: {str(e)}")
        print("Make sure reportlab is installed: pip install reportlab")
        raise

if __name__ == "__main__":
    main()