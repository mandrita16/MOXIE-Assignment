# Methodology

This system evaluates communication skills in tutorial videos using a hybrid approach combining NLP and computer vision.

## NLP Analysis

Transcripts are processed to extract linguistic features:

• Speech rate  
• Filler word frequency  
• Instructional vocabulary  
• Lexical diversity  
• Readability score  
• Topic coherence using TF-IDF  

These features estimate clarity and instructional quality.

## Visual Analysis

OpenCV is used to analyze frames from the video:

• Face detection frequency  
• Visual stability  
• Lighting quality  

These features approximate presenter presence and visual clarity.

## Scoring

All features are normalized and combined into a composite score ranging from **0–10**.