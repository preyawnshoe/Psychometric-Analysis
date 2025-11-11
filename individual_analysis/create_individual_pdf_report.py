"""
Individual Analysis PDF Report Generator
Creates a comprehensive PDF report with tables and visualizations
"""

from reportlab.lib.pagesizes import letter, A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
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
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    # Heading style
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    # Subheading style
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=10,
        spaceBefore=15,
        textColor=colors.darkblue,
        fontName='Helvetica-Bold'
    )
    
    # Body style
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Small text style
    small_style = ParagraphStyle(
        'SmallText',
        parent=styles['Normal'],
        fontSize=8,
        spaceAfter=6,
        alignment=TA_LEFT,
        fontName='Helvetica'
    )
    
    return {
        'title': title_style,
        'heading': heading_style,
        'subheading': subheading_style,
        'body': body_style,
        'small': small_style,
        'normal': styles['Normal']
    }

def load_analysis_data():
    """Load analysis results"""
    base_path = r'C:\Users\p2var\vandana\individual_analysis'
    
    # Load comprehensive profiles
    profiles_path = os.path.join(base_path, 'comprehensive_individual_profiles.csv')
    summary_path = os.path.join(base_path, 'individual_analysis_summary.csv')
    
    data = {}
    
    if os.path.exists(profiles_path):
        data['comprehensive'] = pd.read_csv(profiles_path)
        print(f"✓ Loaded comprehensive profiles: {len(data['comprehensive'])} participants")
    
    if os.path.exists(summary_path):
        data['summary'] = pd.read_csv(summary_path)
        print(f"✓ Loaded summary data: {len(data['summary'])} participants")
    
    return data

def create_cover_page(story, styles):
    """Create the cover page"""
    
    # Title
    story.append(Paragraph("COMPREHENSIVE INDIVIDUAL ANALYSIS REPORT", styles['title']))
    story.append(Spacer(1, 0.5*inch))
    
    # Subtitle
    story.append(Paragraph("Psychological Profiles Integration", styles['heading']))
    story.append(Spacer(1, 0.3*inch))
    
    # Project info
    info_text = """
    <b>Assessment Framework:</b> RIASEC × Goal Orientation × Self-Concept Integration<br/>
    <b>Total Participants:</b> 13 individuals<br/>
    <b>Analysis Date:</b> """ + datetime.now().strftime("%B %d, %Y") + """<br/>
    <b>Assessment Domains:</b> Career Interests, Achievement Motivation, Self-Perception<br/>
    <b>Report Type:</b> Individual Profiling with Psychological Insights
    """
    story.append(Paragraph(info_text, styles['body']))
    story.append(Spacer(1, 1*inch))
    
    # Executive Summary
    story.append(Paragraph("EXECUTIVE SUMMARY", styles['heading']))
    abstract_text = """
    This comprehensive report presents individual psychological profiles for 13 participants, 
    integrating three major assessment domains: RIASEC career interests (Holland's theory), 
    achievement goal orientations (2×2 framework), and multidimensional self-concept measures. 
    Each participant receives a detailed profile including Holland codes, goal orientation 
    classifications, self-concept levels, and personalized insights with development 
    recommendations. The analysis reveals diverse patterns: 38.5% show Investigative interests, 
    38.5% exhibit Performance Approach goals, and 76.9% demonstrate positive to high self-concept 
    levels, indicating a generally adaptive psychological profile across the sample.
    """
    story.append(Paragraph(abstract_text, styles['body']))
    story.append(PageBreak())

def create_overview_section(story, styles, data):
    """Create overview section with summary statistics"""
    
    story.append(Paragraph("1. ANALYSIS OVERVIEW", styles['heading']))
    
    # Methodology
    story.append(Paragraph("1.1 Assessment Framework", styles['subheading']))
    methodology_text = """
    <b>RIASEC Career Interest Assessment:</b> Based on Holland's theory measuring six career 
    interest types (Realistic, Investigative, Artistic, Social, Enterprising, Conventional).<br/><br/>
    
    <b>Achievement Goal Orientation:</b> 2×2 framework assessing four motivation patterns 
    (Performance Approach/Avoidance, Mastery Approach/Avoidance).<br/><br/>
    
    <b>Multidimensional Self-Concept:</b> Ten-domain assessment covering Health/Sex Appropriateness, 
    Abilities, Self-Confidence, Self-Acceptance, Worthiness, Time Perspective, Beliefs/Convictions, 
    Shame/Guilt, Sociability, and Emotional well-being.
    """
    story.append(Paragraph(methodology_text, styles['body']))
    
    # Sample characteristics
    story.append(Paragraph("1.2 Sample Characteristics", styles['subheading']))
    
    if 'summary' in data:
        df = data['summary']
        
        # RIASEC distribution
        riasec_dist = df['riasec_dominant'].value_counts()
        riasec_text = "<b>RIASEC Type Distribution:</b><br/>"
        for riasec_type, count in riasec_dist.items():
            percentage = (count / len(df)) * 100
            riasec_text += f"• {riasec_type}: {count} participants ({percentage:.1f}%)<br/>"
        
        story.append(Paragraph(riasec_text, styles['body']))
        
        # Goal orientation distribution
        goal_dist = df['goal_orientation_type'].value_counts()
        goal_text = "<b>Goal Orientation Patterns:</b><br/>"
        for goal_type, count in goal_dist.items():
            percentage = (count / len(df)) * 100
            goal_text += f"• {goal_type}: {count} participants ({percentage:.1f}%)<br/>"
        
        story.append(Paragraph(goal_text, styles['body']))
        
        # Self-concept distribution
        sc_dist = df['self_concept_type'].value_counts()
        sc_text = "<b>Self-Concept Levels:</b><br/>"
        for sc_type, count in sc_dist.items():
            percentage = (count / len(df)) * 100
            sc_text += f"• {sc_type}: {count} participants ({percentage:.1f}%)<br/>"
        
        story.append(Paragraph(sc_text, styles['body']))
    
    story.append(PageBreak())

def create_summary_table(story, styles, data):
    """Create comprehensive summary table"""
    
    story.append(Paragraph("2. INDIVIDUAL PROFILES SUMMARY", styles['heading']))
    
    if 'summary' in data:
        df = data['summary']
        
        # Prepare table data
        table_data = [
            ['Participant', 'Holland\nCode', 'RIASEC\nDominant', 'Goal\nOrientation', 
             'Self-Concept\nLevel', 'SC\nScore', 'Key Insights']
        ]
        
        for _, row in df.iterrows():
            # Truncate insights for table readability
            insights = str(row.get('key_insights', ''))
            if len(insights) > 60:
                insights = insights[:57] + '...'
            
            table_data.append([
                str(row['participant_title']),
                str(row.get('holland_code', 'N/A')),
                str(row.get('riasec_dominant', 'Unknown')).replace('_', ' '),
                str(row.get('goal_orientation_type', 'Unknown')).replace('-', '\n'),
                str(row.get('self_concept_type', 'Unknown')).replace(' ', '\n'),
                f"{row.get('sc_total', 0):.2f}" if pd.notna(row.get('sc_total')) else 'N/A',
                insights
            ])
        
        # Create table
        summary_table = Table(table_data, colWidths=[1.2*inch, 0.6*inch, 1*inch, 1.2*inch, 1*inch, 0.6*inch, 2*inch])
        
        # Style the table
        summary_table.setStyle(TableStyle([
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            
            # Data styling
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),  # Participant names left-aligned
            ('ALIGN', (-1, 1), (-1, -1), 'LEFT'),  # Insights left-aligned
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            
            # Alternating row colors
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.beige]),
            
            # Grid
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('LINEWIDTH', (0, 0), (-1, 0), 2),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
    
    story.append(PageBreak())

def create_detailed_profiles_section(story, styles, data):
    """Create detailed individual profiles section"""
    
    story.append(Paragraph("3. DETAILED INDIVIDUAL PROFILES", styles['heading']))
    
    if 'summary' in data:
        df = data['summary']
        
        for idx, row in df.iterrows():
            # Individual profile header
            name = row['participant_title']
            story.append(Paragraph(f"3.{idx+1} {name}", styles['subheading']))
            
            # Profile summary table
            profile_data = [
                ['Aspect', 'Details'],
                ['Holland Code', str(row.get('holland_code', 'N/A'))],
                ['Dominant RIASEC', str(row.get('riasec_dominant', 'Unknown'))],
                ['Goal Orientation', str(row.get('goal_orientation_type', 'Unknown'))],
                ['Self-Concept Level', str(row.get('self_concept_type', 'Unknown'))],
                ['Self-Concept Score', f"{row.get('sc_total', 0):.3f}" if pd.notna(row.get('sc_total')) else 'N/A'],
                ['RIASEC Total', f"{row.get('riasec_total', 0):.3f}" if pd.notna(row.get('riasec_total')) else 'N/A'],
                ['Goal Total', f"{row.get('goal_total', 0):.3f}" if pd.notna(row.get('goal_total')) else 'N/A']
            ]
            
            profile_table = Table(profile_data, colWidths=[1.5*inch, 4*inch])
            profile_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(profile_table)
            story.append(Spacer(1, 0.1*inch))
            
            # Insights and recommendations
            insights = str(row.get('key_insights', 'No specific insights available'))
            recommendations = str(row.get('recommendations', 'Continue balanced development'))
            
            story.append(Paragraph("<b>Key Insights:</b>", styles['body']))
            story.append(Paragraph(insights, styles['small']))
            story.append(Spacer(1, 0.1*inch))
            
            story.append(Paragraph("<b>Recommendations:</b>", styles['body']))
            story.append(Paragraph(recommendations, styles['small']))
            story.append(Spacer(1, 0.2*inch))
            
            # Add separator line except for last participant
            if idx < len(df) - 1:
                story.append(Paragraph("─" * 80, styles['small']))
                story.append(Spacer(1, 0.2*inch))
    
    story.append(PageBreak())

def create_analysis_insights_section(story, styles, data):
    """Create analysis insights section"""
    
    story.append(Paragraph("4. ANALYSIS INSIGHTS & PATTERNS", styles['heading']))
    
    # Pattern Analysis
    story.append(Paragraph("4.1 Key Patterns Identified", styles['subheading']))
    
    if 'summary' in data:
        df = data['summary']
        
        patterns_text = """
        <b>RIASEC-Goal Orientation Combinations:</b><br/>
        Analysis reveals several interesting patterns in career interest and motivation combinations:
        <br/><br/>
        """
        
        # Analyze combinations
        investigative_mastery = len(df[(df['riasec_dominant'] == 'Investigative') & 
                                     (df['goal_orientation_type'].str.contains('Mastery', na=False))])
        realistic_performance = len(df[(df['riasec_dominant'] == 'Realistic') & 
                                     (df['goal_orientation_type'].str.contains('Performance', na=False))])
        
        patterns_text += f"• {investigative_mastery} Investigative types show Mastery orientation (natural researchers)<br/>"
        patterns_text += f"• {realistic_performance} Realistic types show Performance orientation (practical achievers)<br/><br/>"
        
        # Self-concept patterns
        high_sc_count = len(df[df['self_concept_type'].str.contains('High', na=False)])
        good_sc_count = len(df[df['self_concept_type'].str.contains('Good', na=False)])
        
        patterns_text += f"<b>Self-Concept Distribution:</b><br/>"
        patterns_text += f"• {high_sc_count} participants show High self-concept levels<br/>"
        patterns_text += f"• {good_sc_count} participants show Good self-concept levels<br/>"
        patterns_text += f"• Overall positive self-perception indicates healthy psychological adjustment<br/>"
        
        story.append(Paragraph(patterns_text, styles['body']))
    
    # Recommendations
    story.append(Paragraph("4.2 Development Recommendations", styles['subheading']))
    
    recommendations_text = """
    <b>Individual Development Strategies:</b><br/><br/>
    
    <b>For Investigative Types:</b> Encourage research projects, independent study, and 
    analytical problem-solving activities to leverage natural curiosity and systematic thinking.<br/><br/>
    
    <b>For Realistic Types:</b> Provide hands-on learning opportunities, practical applications, 
    and experiential learning to match their concrete, action-oriented preferences.<br/><br/>
    
    <b>For Performance-Oriented Individuals:</b> Set clear achievement targets, provide competitive 
    environments, and offer recognition systems to motivate optimal performance.<br/><br/>
    
    <b>For Mastery-Oriented Individuals:</b> Focus on deep understanding, skill development, 
    and learning process rather than just outcomes to maintain intrinsic motivation.<br/><br/>
    
    <b>For High Self-Concept Individuals:</b> Leverage confidence for leadership roles and 
    challenging goals while maintaining balanced self-awareness.<br/><br/>
    
    <b>For Moderate Self-Concept Individuals:</b> Provide supportive environments and 
    strength-building activities to enhance positive self-perception.
    """
    
    story.append(Paragraph(recommendations_text, styles['body']))
    story.append(PageBreak())

def create_methodology_appendix(story, styles, data):
    """Create methodology appendix"""
    
    story.append(Paragraph("APPENDIX: METHODOLOGY & DATA SOURCES", styles['heading']))
    
    # Data sources
    story.append(Paragraph("A.1 Data Sources Integration", styles['subheading']))
    
    methodology_text = """
    This comprehensive analysis integrates data from multiple assessment sources:
    <br/><br/>
    
    <b>Primary Datasets:</b><br/>
    • RIASEC-Goal Orientation Analysis (14 participants)<br/>
    • RIASEC-Self Concept Analysis (12 participants)<br/>
    • Goal-Self Concept Analysis (13 participants)<br/>
    • Individual Statistical Profiles (14 participants)<br/><br/>
    
    <b>Data Processing:</b><br/>
    • Participant name standardization across all datasets<br/>
    • Missing data handling using available information<br/>
    • Cross-dataset validation and consistency checks<br/>
    • Statistical profile calculation and normalization<br/><br/>
    
    <b>Classification Systems:</b><br/>
    • Holland Codes: Top 3 RIASEC types for each participant<br/>
    • Goal Orientation Types: Based on dominant achievement goal pattern<br/>
    • Self-Concept Levels: Quartile-based classification system<br/>
    • Psychological Insights: Pattern-based inference system
    """
    
    story.append(Paragraph(methodology_text, styles['body']))
    
    # Technical details
    story.append(Paragraph("A.2 Technical Specifications", styles['subheading']))
    
    technical_text = """
    <b>Statistical Analysis:</b><br/>
    • Correlation analysis between assessment domains<br/>
    • Profile consistency calculations<br/>
    • Approach vs. avoidance ratio computations<br/>
    • Performance vs. mastery orientation indices<br/><br/>
    
    <b>Classification Criteria:</b><br/>
    • High Self-Concept: ≥ 3.5 total score<br/>
    • Good Self-Concept: 3.0 - 3.49 total score<br/>
    • Moderate Self-Concept: 2.5 - 2.99 total score<br/>
    • Low Self-Concept: < 2.5 total score<br/><br/>
    
    <b>Report Generation:</b><br/>
    • Automated profile synthesis from multiple data sources<br/>
    • Pattern recognition for psychological insights<br/>
    • Evidence-based recommendation generation<br/>
    • Quality assurance through cross-validation
    """
    
    story.append(Paragraph(technical_text, styles['body']))

def main():
    """Main PDF report generation function"""
    print("CREATING COMPREHENSIVE INDIVIDUAL ANALYSIS PDF REPORT")
    print("=" * 60)
    
    # Load data
    print("Loading analysis data...")
    data = load_analysis_data()
    
    if not data:
        print("❌ No data available for report generation!")
        return
    
    # Create PDF document
    output_path = r'C:\Users\p2var\vandana\individual_analysis\Individual_Analysis_Report.pdf'
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Create styles
    styles = create_custom_styles()
    
    # Build story
    story = []
    
    print("Creating cover page...")
    create_cover_page(story, styles)
    
    print("Creating overview section...")
    create_overview_section(story, styles, data)
    
    print("Creating summary table...")
    create_summary_table(story, styles, data)
    
    print("Creating detailed profiles...")
    create_detailed_profiles_section(story, styles, data)
    
    print("Creating insights section...")
    create_analysis_insights_section(story, styles, data)
    
    print("Creating methodology appendix...")
    create_methodology_appendix(story, styles, data)
    
    # Build PDF
    print("Building PDF document...")
    doc.build(story)
    
    print(f"\n✅ PDF report created successfully!")
    print(f"📄 Location: {output_path}")
    
    if 'summary' in data:
        print(f"\n📊 REPORT SUMMARY:")
        print(f"- Total Participants: {len(data['summary'])}")
        print(f"- Assessment Domains: 3 (RIASEC, Goal Orientation, Self-Concept)")
        print(f"- Individual Profiles: Complete psychological analysis")
        print(f"- Tables: Summary table + detailed profile tables")
        print(f"- Insights: Pattern analysis + personalized recommendations")
    
    print("\n🎯 PDF Report Features:")
    print("• Executive summary with key findings")
    print("• Comprehensive summary table with all participants")
    print("• Detailed individual profile pages")
    print("• Pattern analysis and development insights")
    print("• Methodology appendix with technical details")

if __name__ == "__main__":
    main()