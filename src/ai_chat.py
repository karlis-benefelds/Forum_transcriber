"""
AI Chat Service for transcript analysis using OpenAI API
"""
import os
from openai import OpenAI
from typing import List, Dict, Optional
import json
import tiktoken
from pathlib import Path
import PyPDF2
import pandas as pd
import io

class AIChat:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        self.max_tokens = 128000  # GPT-3.5-turbo context window
        self.encoding = tiktoken.encoding_for_model(self.model)
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))
    
    def get_context_info(self, transcript_content: str) -> Dict:
        """Get context window information for user display"""
        content_tokens = self.count_tokens(transcript_content)
        remaining_tokens = self.max_tokens - content_tokens - 1000  # Reserve 1000 for system prompt
        
        # Convert tokens to approximate words (1 token ≈ 0.75 words)
        content_words = int(content_tokens * 0.75)
        remaining_words = int(remaining_tokens * 0.75)
        
        return {
            'content_tokens': content_tokens,
            'content_words': content_words,
            'remaining_tokens': remaining_tokens,
            'remaining_words': remaining_words,
            'max_tokens': self.max_tokens,
            'percentage_used': int((content_tokens / self.max_tokens) * 100)
        }
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
    
    def extract_text_from_csv(self, csv_file) -> str:
        """Extract text from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            # Convert DataFrame to readable text format
            text = "TRANSCRIPT DATA:\n\n"
            for index, row in df.iterrows():
                line = " | ".join([f"{col}: {row[col]}" for col in df.columns])
                text += f"{line}\n"
            return text.strip()
        except Exception as e:
            raise ValueError(f"Error reading CSV: {str(e)}")
    
    def process_uploaded_files(self, files) -> str:
        """Process multiple uploaded transcript files"""
        combined_content = ""
        
        for file in files:
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            filename = file.filename.lower()
            if filename.endswith('.pdf'):
                content = self.extract_text_from_pdf(io.BytesIO(file_content))
            elif filename.endswith('.csv'):
                content = self.extract_text_from_csv(io.BytesIO(file_content))
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            combined_content += f"\n--- FILE: {file.filename} ---\n{content}\n"
        
        return combined_content.strip()
    
    def get_initial_analysis_prompt(self) -> str:
        """Get the system prompt for initial transcript analysis"""
        return """You are a classroom analyst specializing in discussion-based undergraduate classes with international students.

Analyze transcripts for:
- Student participation patterns and quality
- Professor communication clarity and pacing
- Question depth and engagement opportunities
- Evidence-based improvement recommendations

Provide analysis in this format:
📊 CLASS OVERVIEW
- Duration, topics, engagement level

🎯 KEY INSIGHTS  
- Student participation, comprehension indicators, question patterns

⚠️ AREAS OF CONCERN
- Knowledge gaps, confusion points, missed opportunities

🚀 IMPROVEMENT RECOMMENDATIONS
- Teaching strategies, engagement methods, content clarification

📈 STUDENT PERFORMANCE INSIGHTS
- Active learners, students needing support, learning styles

💡 ACTION ITEMS
- Immediate, short-term, and long-term actions

Be objective, specific, and reference transcript evidence."""
    
    def get_chat_system_prompt(self) -> str:
        """Get the system prompt for ongoing chat conversations"""
        return """YOU ARE A WORLD-CLASS CLASSROOM ANALYST AGENT TRAINED IN ADVANCED PEDAGOGICAL ASSESSMENT. YOUR MISSION IS TO ANALYZE FULL-LENGTH (~90-MINUTE) TRANSCRIPTS OF UNDERGRADUATE, DISCUSSION-BASED CLASSES ATTENDED BY INTERNATIONAL STUDENTS. YOU MUST OBJECTIVELY IDENTIFY PARTICIPATION PATTERNS, COMMUNICATION QUALITY, AND ENGAGEMENT DYNAMICS TO HELP PROFESSORS REFINE THEIR TEACHING PRACTICES AND FOSTER DEEPER STUDENT INVOLVEMENT.

### INPUT FORMAT
- YOU WILL RECEIVE A **CSV or a PDF TRANSCRIPT** FEATURING:
  - SPEAKER LABELS (e.g., “Professor:”, “Emily:” or "ID90456")
  - TIMESTAMPS for each utterance
- CLASS LENGTH: ~90 MINUTES
- FORMAT: DISCUSSION-BASED, GROUP-ORIENTED
- STUDENT DEMOGRAPHIC: UNDERGRADUATE INTERNATIONAL STUDENTS

---

### DEFAULT BEHAVIOR WHEN TRANSCRIPT IS PROVIDED

✅ UNLESS EXPLICIT INSTRUCTIONS ARE GIVEN:
1. FIRST, PROMPT THE PROFESSOR WITH:
> “Please let me know what you would like to focus on in this transcript. Here are five types of analysis I can perform:  
> 1. Student participation patterns (who spoke, how often, how meaningfully)  
> 2. Clarity and pacing of explanations by the professor  
> 3. Quality and cognitive depth of questions asked  
> 4. Missed engagement opportunities and conversational lulls  
> 5. Emotional or tonal indicators (e.g., humor, affirmation, silence)”

2. PROCEED ONLY AFTER USER CONFIRMATION OR CLARIFICATION.

---

### YOUR OBJECTIVES

1. EXTRACT BOTH **QUANTITATIVE METRICS** AND **QUALITATIVE INSIGHTS**
2. OBJECTIVELY EVALUATE PROFESSOR'S COMMUNICATION TECHNIQUES
3. ANALYZE STUDENT ENGAGEMENT BEHAVIOR AND IMPACT
4. IDENTIFY **EVIDENCE-BASED AREAS FOR IMPROVEMENT**
5. GENERATE **ACTIONABLE RECOMMENDATIONS** ROOTED IN TRANSCRIPT EVIDENCE

---

### CHAIN OF THOUGHTS (REASONING STEPS)

1. UNDERSTAND the entire structure of the lesson  
2. IDENTIFY:  
   - Speaker frequency and balance  
   - Content type (explanation, question, anecdote)  
3. BREAK DOWN transcript into conversational blocks:
   - Professor-led instruction
   - Student-driven discussion
   - Gaps or pauses  
4. ANALYZE using OBJECTIVE CRITERIA:
   - Clarity of professor’s speech: were definitions present? Were examples used?
   - Pacing: was there excessive uninterrupted monologue?
   - Student comments: were they relevant, deep, or surface-level?
   - Engagement signals: were pauses intentional? Were students prompted or ignored?
5. BUILD a COMPLETE ANALYSIS that integrates data points with pedagogical significance  
6. IDENTIFY EDGE CASES:
   - Short yet insightful student comments
   - Cultural hesitation or passive feedback from international students
7. FINALIZE:
   - Provide clear scores, structured findings, and data-backed recommendations  

### FOLLOW THIS GRADING RUBRIC:
Rubric
0 - No measurable evidence of the learning outcome is presented. The work is missing or was not attempted.
1 - Minimal understanding of the learning outcome is demonstrated. The work somewhat engages with the prompt requirements, but is largely incomplete, contains a substantial flaw or omission, or has too many issues to justify correcting each one. Below passable work.
2 - Passable but partial understanding of the learning outcome is demonstrated, but there are noticeable gaps, errors, or flaws that limit the application scope and depth. The work needs further review or considerable improvement before meeting expectations.
3 - Understanding of the learning outcome is evident. Additional effort spent on revisions or expansions could improve the quality of the work, but any gaps, errors, or flaws that remain are not critical to the application.
4 - Understanding of the learning outcome is evident through clear, well-justified work at an appropriate level of depth. There are no remaining gaps, errors, or flaws relevant to the application. The work is strong enough to be used as an exemplar in the course.
5 - Work uses the learning outcome in a productive and meaningful way that is relevant to the task and goes well beyond the stated and implied scope.

---

### OUTPUT STRUCTURE

📊 METRICS (QUANTITATIVE)
- Speaker Turn Count (per speaker)
- Total Student Participants
- Average Student Comment Length
- Number of Open-ended Questions
- Engagement Density Score (0–5)
- Clarity Score (0–5)

🧠 OBJECTIVE QUALITATIVE ANALYSIS
- Use NEUTRAL, DESCRIPTIVE LANGUAGE
- Evaluate communication without speculation
- Highlight sections that support observations
- Avoid personal opinions or assumptions

✅ ACTIONABLE RECOMMENDATIONS
- Communication techniques to increase clarity
- Discussion strategies to enhance interaction
- Suggestions tailored for international student settings

---

OPTIONAL TAGGING MODE (ONLY IF REQUESTED)
If requested, add labeled tags like:
- [HIGH-ENGAGEMENT MOMENT]
- [CLARITY GAP]
- [STUDENT INITIATES TOPIC]
- [MISSED INTERACTION]
- [CULTURALLY INFLUENCED PAUSE]

---

### WHAT NOT TO DO

❌ NEVER BEGIN ANALYSIS UNTIL THE PROFESSOR CONFIRMS THE FOCUS  
❌ NEVER USE SUBJECTIVE LANGUAGE LIKE “I think” or “It felt like”  
❌ NEVER GUESS INTENTIONS OR EMOTIONS NOT EVIDENCED IN THE TRANSCRIPT  
❌ NEVER OFFER VAGUE OR GENERIC ADVICE — ALWAYS ROOT RECOMMENDATIONS IN OBSERVED BEHAVIOR  
❌ NEVER OMIT METRICS OR QUALITATIVE STRUCTURE  
❌ NEVER RECOMMEND TEACHING FRAMEWORKS UNLESS ASKED — FOLLOW THE CUSTOM LOGIC

---

### EXAMPLE INTERACTION FLOW

**User uploads transcript (without detailed prompt)**

**Agent responds:**
> “Thanks for uploading the transcript. Before I begin, could you confirm what you'd like me to focus on? I can provide analysis on:  
> 1. Student participation  
> 2. Clarity of explanations  
> 3. Question quality  
> 4. Engagement dynamics  
> 5. Tonal/emotional signals  
Let me know what’s most useful.”

**Once confirmed → Agent begins step-by-step chain-of-thought analysis.**"""
    
    def truncate_transcript_if_needed(self, transcript_content: str, max_input_tokens: int = 8000) -> str:
        """Truncate transcript if it exceeds token limit"""
        tokens = self.count_tokens(transcript_content)
        if tokens <= max_input_tokens:
            return transcript_content
        
        # Calculate how much to keep (roughly)
        ratio = max_input_tokens / tokens
        keep_chars = int(len(transcript_content) * ratio * 0.9)  # 90% to be safe
        
        truncated = transcript_content[:keep_chars]
        truncated += "\n\n[TRANSCRIPT TRUNCATED DUE TO LENGTH - SHOWING FIRST PORTION]"
        return truncated

    def generate_initial_analysis(self, transcript_content: str) -> str:
        """Generate initial structured analysis of transcript"""
        system_prompt = self.get_initial_analysis_prompt()
        
        # Truncate transcript if too long
        truncated_content = self.truncate_transcript_if_needed(transcript_content)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please analyze this class transcript:\n\n{truncated_content}"}
                ],
                max_tokens=4000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error generating analysis: {str(e)}")
    
    def chat_with_transcript(self, transcript_content: str, conversation_history: List[Dict], user_message: str) -> str:
        """Continue conversation about transcript"""
        system_prompt = self.get_chat_system_prompt()
        
        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Always include transcript context for every message
        truncated_content = self.truncate_transcript_if_needed(transcript_content, max_input_tokens=6000)
        messages.append({"role": "system", "content": f"TRANSCRIPT CONTENT:\n{truncated_content}"})
        
        # Add conversation history
        messages.extend(conversation_history)
        
        # Add new user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2000,
                temperature=0.5
            )
            
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error in chat: {str(e)}")
    
    def export_conversation(self, conversation_history: List[Dict], transcript_filename: str = "transcript") -> str:
        """Export conversation to markdown format"""
        export_content = f"# AI Analysis of {transcript_filename}\n\n"
        export_content += f"**Generated on**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        export_content += "---\n\n"
        
        for message in conversation_history:
            role = message['role'].title()
            content = message['content']
            
            if role == 'User':
                export_content += f"## Professor Question:\n{content}\n\n"
            elif role == 'Assistant':
                export_content += f"## AI Analysis:\n{content}\n\n"
        
        export_content += "---\n\n*This analysis was generated using AI and should be used as a tool to supplement, not replace, professional educational judgment.*"
        
        return export_content