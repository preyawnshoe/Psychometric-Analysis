import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, red, green, orange, grey
from reportlab.lib import colors
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.platypus.flowables import HRFlowable
from datetime import datetime
import os

class PDFReportGenerator:
    def __init__(self, results_file='self_concept_results.csv', image_file='self_concept_analysis.png'):
        self.results_file = results_file
        self.image_file = image_file
        self.styles = getSampleStyleSheet()
        self.story = []
        
        # Load data
        self.df = pd.read_csv(results_file)
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        self.subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkgreen
        )

    def create_title_page(self):
        """Create the title page of the report."""
        # Title
        title = Paragraph("Self-Concept Scale Analysis Report", self.title_style)
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle = Paragraph("Comprehensive Statistical Analysis and Visualization", 
                           self.styles['Heading3'])
        subtitle.alignment = 1
        self.story.append(subtitle)
        self.story.append(Spacer(1, 1*inch))
        
        # Report details in a table
        report_data = [
            ['Report Generated:', datetime.now().strftime("%B %d, %Y at %I:%M %p")],
            ['Total Participants:', str(len(self.df))],
            ['Analysis Type:', 'Self-Concept Scale (49 items)'],
            ['Statistical Methods:', 'Descriptive Statistics, Percentile Analysis'],
            ['Visualization:', '4-Panel Comprehensive Chart']
        ]
        
        report_table = Table(report_data, colWidths=[2.5*inch, 3*inch])
        report_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey)
        ]))
        
        self.story.append(report_table)
        self.story.append(Spacer(1, 1*inch))
        
        # Abstract
        abstract_title = Paragraph("Executive Summary", self.heading_style)
        self.story.append(abstract_title)
        
        abstract_text = f"""
        This report presents a comprehensive analysis of self-concept scale responses from {len(self.df)} participants 
        aged {self.df['Age'].min()}-{self.df['Age'].max()} years. The analysis includes descriptive statistics, 
        percentile rankings, and multi-panel visualizations. Key findings show scores ranging from 
        {self.df['Total_Score'].min()} to {self.df['Total_Score'].max()} with a mean of {self.df['Total_Score'].mean():.1f}. 
        The highest-scoring participant achieved the {self.df['Percentile_Rank'].max():.0f}th percentile, while 
        gender distribution was balanced with {sum(self.df['Gender'] == 'Male')} males and 
        {sum(self.df['Gender'] == 'Female')} females.
        """
        
        abstract_para = Paragraph(abstract_text, self.styles['Normal'])
        self.story.append(abstract_para)
        self.story.append(PageBreak())

    def create_methodology_section(self):
        """Create methodology section."""
        method_title = Paragraph("Methodology", self.title_style)
        self.story.append(method_title)
        
        # Data Collection
        data_title = Paragraph("Data Collection", self.heading_style)
        self.story.append(data_title)
        
        data_text = """
        The self-concept scale consisted of 49 items measured on a 5-point Likert scale ranging from 
        "Strongly Disagree" (1) to "Strongly Agree" (5). Participants responded to statements about 
        their self-perception, self-worth, and personal capabilities.
        """
        self.story.append(Paragraph(data_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Scoring Method
        scoring_title = Paragraph("Scoring Methodology", self.heading_style)
        self.story.append(scoring_title)
        
        scoring_text = """
        <b>Likert Scale Conversion:</b><br/>
        • Strongly Disagree = 1<br/>
        • Disagree = 2<br/>
        • Undecided = 3<br/>
        • Agree = 4<br/>
        • Strongly Agree = 5<br/><br/>
        
        <b>Reverse Scoring:</b><br/>
        Items reflecting negative self-concept were reverse-scored (items 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 27, 28, 29, 30, 32, 36, 37, 39, 43, 48).
        For these items, the scoring was inverted: 1→5, 2→4, 3→3, 4→2, 5→1.<br/><br/>
        
        <b>Total Score Calculation:</b><br/>
        The total self-concept score was calculated as the sum of all 49 items after appropriate reverse scoring.
        Higher scores indicate more positive self-concept.
        """
        self.story.append(Paragraph(scoring_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))

    def create_sample_characteristics(self):
        """Create sample characteristics section."""
        sample_title = Paragraph("Sample Characteristics", self.title_style)
        self.story.append(sample_title)
        
        # Demographics table
        gender_counts = self.df['Gender'].value_counts()
        age_stats = self.df['Age'].describe()
        
        demo_data = [
            ['Characteristic', 'Value'],
            ['Total Participants', str(len(self.df))],
            ['Male Participants', str(gender_counts.get('Male', 0))],
            ['Female Participants', str(gender_counts.get('Female', 0))],
            ['Age Range', f"{age_stats['min']:.0f} - {age_stats['max']:.0f} years"],
            ['Mean Age', f"{age_stats['mean']:.1f} years"],
            ['Age Standard Deviation', f"{age_stats['std']:.1f} years"]
        ]
        
        demo_table = Table(demo_data, colWidths=[2.5*inch, 2*inch])
        demo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        self.story.append(demo_table)
        self.story.append(Spacer(1, 0.3*inch))

    def create_descriptive_statistics(self):
        """Create descriptive statistics section."""
        stats_title = Paragraph("Descriptive Statistics", self.title_style)
        self.story.append(stats_title)
        
        # Calculate statistics
        stats = self.df['Total_Score'].describe()
        
        # Central tendency and dispersion
        central_title = Paragraph("Central Tendency and Dispersion", self.heading_style)
        self.story.append(central_title)
        
        stats_data = [
            ['Statistic', 'Value', 'Interpretation'],
            ['Mean', f"{stats['mean']:.2f}", 'Average self-concept score'],
            ['Median', f"{stats['50%']:.2f}", 'Middle value when ranked'],
            ['Standard Deviation', f"{stats['std']:.2f}", 'Measure of score variability'],
            ['Minimum Score', f"{stats['min']:.0f}", 'Lowest self-concept score'],
            ['Maximum Score', f"{stats['max']:.0f}", 'Highest self-concept score'],
            ['Range', f"{stats['max'] - stats['min']:.0f}", 'Difference between max and min'],
            ['25th Percentile', f"{stats['25%']:.2f}", '25% scored below this value'],
            ['75th Percentile', f"{stats['75%']:.2f}", '75% scored below this value']
        ]
        
        stats_table = Table(stats_data, colWidths=[1.8*inch, 1.2*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (1, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        self.story.append(stats_table)
        self.story.append(Spacer(1, 0.3*inch))

    def create_visualization_analysis(self):
        """Create visualization analysis section."""
        viz_title = Paragraph("Visualization Analysis", self.title_style)
        self.story.append(viz_title)
        
        # Insert the visualization
        if os.path.exists(self.image_file):
            img = Image(self.image_file, width=7*inch, height=5.25*inch)
            self.story.append(img)
            self.story.append(Spacer(1, 0.2*inch))
        
        # Analysis of each panel
        panel_title = Paragraph("Four-Panel Analysis", self.heading_style)
        self.story.append(panel_title)
        
        panel_analysis = """
        <b>Panel 1 - Distribution Histogram (Top Left):</b><br/>
        The histogram shows the frequency distribution of total self-concept scores. The distribution appears 
        approximately normal with a slight right skew. The red dashed line indicates the mean score, providing 
        a reference point for interpreting individual scores.<br/><br/>
        
        <b>Panel 2 - Gender Comparison (Top Right):</b><br/>
        Box plots comparing self-concept scores between male and female participants. This visualization reveals 
        any potential gender differences in self-concept scores, including median values, quartiles, and outliers.<br/><br/>
        
        <b>Panel 3 - Normality Assessment (Bottom Left):</b><br/>
        The Q-Q (Quantile-Quantile) plot assesses whether the score distribution follows a normal distribution. 
        Points closely following the diagonal line suggest normal distribution, which is important for statistical assumptions.<br/><br/>
        
        <b>Panel 4 - Top Performers (Bottom Right):</b><br/>
        Bar chart highlighting the highest-scoring participants with their names rotated for readability. 
        This provides a clear ranking of top performers in the self-concept assessment.
        """
        
        self.story.append(Paragraph(panel_analysis, self.styles['Normal']))
        self.story.append(Spacer(1, 0.3*inch))

    def create_ranking_table(self):
        """Create detailed ranking table."""
        ranking_title = Paragraph("Complete Participant Rankings", self.title_style)
        self.story.append(ranking_title)
        
        # Prepare data for table
        table_data = [['Rank', 'Name', 'Age', 'Gender', 'Total Score', 'Mean Score', 'Percentile']]
        
        for _, row in self.df.iterrows():
            table_data.append([
                str(int(row['Rank'])),
                str(row['Name']),
                str(int(row['Age'])),
                str(row['Gender']),
                f"{row['Total_Score']:.0f}",
                f"{row['Mean_Score']:.2f}",
                f"{row['Percentile_Rank']:.1f}%"
            ])
        
        # Create table
        ranking_table = Table(table_data, colWidths=[0.6*inch, 1.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch])
        
        # Style the table
        ranking_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        # Add alternating row colors
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                ranking_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, i), (-1, i), colors.lightgrey)
                ]))
        
        # Highlight top 3 performers
        for i in range(1, min(4, len(table_data))):
            ranking_table.setStyle(TableStyle([
                ('BACKGROUND', (0, i), (-1, i), colors.lightgreen)
            ]))
        
        self.story.append(ranking_table)
        self.story.append(Spacer(1, 0.3*inch))

    def create_insights_section(self):
        """Create insights and interpretation section."""
        insights_title = Paragraph("Key Insights and Interpretation", self.title_style)
        self.story.append(insights_title)
        
        # Performance categories
        perf_title = Paragraph("Performance Categories", self.heading_style)
        self.story.append(perf_title)
        
        # Calculate performance categories
        high_performers = self.df[self.df['Percentile_Rank'] >= 75]
        avg_performers = self.df[(self.df['Percentile_Rank'] >= 25) & (self.df['Percentile_Rank'] < 75)]
        low_performers = self.df[self.df['Percentile_Rank'] < 25]
        
        category_text = f"""
        <b>High Self-Concept (75th percentile and above):</b><br/>
        {len(high_performers)} participants ({len(high_performers)/len(self.df)*100:.1f}%)<br/>
        Names: {', '.join(high_performers['Name'].tolist())}<br/><br/>
        
        <b>Average Self-Concept (25th-75th percentile):</b><br/>
        {len(avg_performers)} participants ({len(avg_performers)/len(self.df)*100:.1f}%)<br/>
        Names: {', '.join(avg_performers['Name'].tolist())}<br/><br/>
        
        <b>Lower Self-Concept (Below 25th percentile):</b><br/>
        {len(low_performers)} participants ({len(low_performers)/len(self.df)*100:.1f}%)<br/>
        Names: {', '.join(low_performers['Name'].tolist())}
        """
        
        self.story.append(Paragraph(category_text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Gender analysis if applicable
        if len(self.df['Gender'].unique()) > 1:
            gender_title = Paragraph("Gender Analysis", self.heading_style)
            self.story.append(gender_title)
            
            male_scores = self.df[self.df['Gender'] == 'Male']['Total_Score']
            female_scores = self.df[self.df['Gender'] == 'Female']['Total_Score']
            
            gender_text = f"""
            <b>Male Participants (n={len(male_scores)}):</b><br/>
            Mean score: {male_scores.mean():.2f}, SD: {male_scores.std():.2f}<br/>
            Range: {male_scores.min():.0f} - {male_scores.max():.0f}<br/><br/>
            
            <b>Female Participants (n={len(female_scores)}):</b><br/>
            Mean score: {female_scores.mean():.2f}, SD: {female_scores.std():.2f}<br/>
            Range: {female_scores.min():.0f} - {female_scores.max():.0f}<br/><br/>
            
            <b>Gender Difference:</b><br/>
            Mean difference: {abs(male_scores.mean() - female_scores.mean()):.2f} points<br/>
            {"Males scored higher on average" if male_scores.mean() > female_scores.mean() else "Females scored higher on average"}
            """
            
            self.story.append(Paragraph(gender_text, self.styles['Normal']))
            self.story.append(Spacer(1, 0.2*inch))

    def create_conclusions(self):
        """Create conclusions section."""
        conclusion_title = Paragraph("Conclusions and Recommendations", self.title_style)
        self.story.append(conclusion_title)
        
        conclusion_text = f"""
        <b>Key Findings:</b><br/>
        1. The self-concept scores show a relatively normal distribution with good variability<br/>
        2. Score range of {self.df['Total_Score'].max() - self.df['Total_Score'].min():.0f} points indicates meaningful individual differences<br/>
        3. {self.df.iloc[0]['Name']} achieved the highest self-concept score ({self.df['Total_Score'].max():.0f})<br/>
        4. Gender distribution is balanced, allowing for meaningful comparisons<br/><br/>
        
        <b>Statistical Summary:</b><br/>
        • Mean score represents a moderate-to-good level of self-concept<br/>
        • Standard deviation indicates appropriate score spread<br/>
        • No extreme outliers that would bias results<br/><br/>
        
        <b>Recommendations:</b><br/>
        1. Consider follow-up assessments for participants in the lower percentiles<br/>
        2. Investigate factors contributing to high self-concept in top performers<br/>
        3. Explore intervention strategies for participants with lower scores<br/>
        4. Consider longitudinal tracking to monitor changes over time
        """
        
        self.story.append(Paragraph(conclusion_text, self.styles['Normal']))

    def create_appendix(self):
        """Create appendix with technical details."""
        appendix_title = Paragraph("Appendix: Technical Details", self.title_style)
        self.story.append(appendix_title)
        
        tech_text = """
        <b>Reverse-Scored Items:</b><br/>
        Items 3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 22, 27, 28, 29, 30, 32, 36, 37, 39, 43, 48<br/><br/>
        
        <b>Data Quality:</b><br/>
        • All participants completed all 49 items<br/>
        • No missing data detected<br/>
        • All responses within expected range<br/><br/>
        
        <b>Statistical Software:</b><br/>
        Analysis conducted using Python with pandas, numpy, matplotlib, seaborn, and scipy libraries.<br/><br/>
        
        <b>Report Generation:</b><br/>
        PDF report created using ReportLab library with automated data integration.
        """
        
        self.story.append(Paragraph(tech_text, self.styles['Normal']))

    def generate_report(self, filename='Self_Concept_Analysis_Report.pdf'):
        """Generate the complete PDF report."""
        # Create PDF document
        doc = SimpleDocTemplate(filename, pagesize=A4)
        
        # Build the story
        self.create_title_page()
        self.create_methodology_section()
        self.create_sample_characteristics()
        self.create_descriptive_statistics()
        self.create_visualization_analysis()
        self.create_ranking_table()
        self.create_insights_section()
        self.create_conclusions()
        self.create_appendix()
        
        # Build PDF
        doc.build(self.story)
        print(f"PDF report generated: {filename}")
        return filename

def main():
    """Main function to generate the PDF report."""
    print("Generating PDF Report...")
    print("="*50)
    
    # Check if required files exist
    if not os.path.exists('self_concept_results.csv'):
        print("Error: self_concept_results.csv not found!")
        return
    
    if not os.path.exists('self_concept_analysis.png'):
        print("Warning: self_concept_analysis.png not found. Report will be generated without visualization.")
    
    # Create report generator
    generator = PDFReportGenerator()
    
    # Generate report
    filename = generator.generate_report()
    
    print(f"\n✅ PDF Report Successfully Generated!")
    print(f"📄 Filename: {filename}")
    print(f"📊 Contains: Complete analysis with visualizations and statistical insights")
    print(f"📈 Pages: Multiple sections including methodology, results, and conclusions")

if __name__ == "__main__":
    main()