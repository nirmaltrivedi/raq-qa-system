import re
import unicodedata
from typing import Dict
from app.core.logging import app_logger as logger
import time


class TextCleaner:
    
    @staticmethod
    def stage_1_structural_cleaning(text: str) -> str:
        logger.debug("Stage 1: Structural cleaning")
        
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        lines = text.split('\n')
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line:
                if (i + 1 < len(lines) and 
                    line and 
                    not line[-1] in '.!?:;' and 
                    lines[i + 1].strip() and 
                    not lines[i + 1].strip()[0].isupper()):
                    line = line + ' ' + lines[i + 1].strip()
                    i += 2
                else:
                    fixed_lines.append(line)
                    i += 1
            else:
                i += 1
        
        text = '\n'.join(fixed_lines)
        
        text = re.sub(r'(?i)^\s*page\s+\d+\s*(?:of\s+\d+)?\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'(?i)^\s*\d+\s*$', '', text, flags=re.MULTILINE)  # Standalone page numbers
        
        return text
    
    @staticmethod
    def stage_2_whitespace_normalization(text: str) -> str:
        logger.debug("Stage 2: Whitespace normalization")
        
        text = re.sub(r' +', ' ', text)
        
        text = text.replace('\t', '  ')
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        lines = [line for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        text = text.strip()
        
        return text
    
    @staticmethod
    def stage_3_character_cleaning(text: str) -> str:
        logger.debug("Stage 3: Character cleaning")
        
        text = unicodedata.normalize('NFC', text)
        
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C' or char in '\n\t')
        
        text = text.replace('"', '"').replace('"', '"')  
        text = text.replace(''', "'").replace(''', "'")  
        text = text.replace('‚', "'").replace('„', '"') 
        
        text = text.replace('—', '-').replace('–', '-')
        
        text = text.replace('\u200b', '')  
        text = text.replace('\u200c', '')  
        text = text.replace('\u200d', '')
        text = text.replace('\ufeff', '') 
        
        text = text.replace('\u00ad', '')
        
        return text
    
    @staticmethod
    def stage_4_content_normalization(text: str) -> str:
        logger.debug("Stage 4: Content normalization")
        
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        text = re.sub(r'\.{4,}', '...', text)  
        text = re.sub(r'!{2,}', '!', text)  
        text = re.sub(r'\?{2,}', '?', text) 
        
        def normalize_email(match):
            return match.group(0).lower()
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', normalize_email, text)
        
        text = re.sub(r'\(\s+', '(', text)  
        text = re.sub(r'\s+\)', ')', text)  
        
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        return text
    
    @classmethod
    def clean_text(cls, text: str) -> Dict:
        start_time = time.time()
        original_length = len(text)
        
        logger.info(f"Starting text cleaning pipeline (original length: {original_length} chars)")
        
        text = cls.stage_1_structural_cleaning(text)
        stage1_length = len(text)
        
        text = cls.stage_2_whitespace_normalization(text)
        stage2_length = len(text)
        
        text = cls.stage_3_character_cleaning(text)
        stage3_length = len(text)
        
        text = cls.stage_4_content_normalization(text)
        final_length = len(text)
        
        processing_time = time.time() - start_time
        
        result = {
            "cleaned_text": text,
            "metadata": {
                "original_char_count": original_length,
                "cleaned_char_count": final_length,
                "chars_removed": original_length - final_length,
                "reduction_percentage": round(((original_length - final_length) / original_length * 100), 2) if original_length > 0 else 0,
                "processing_time": round(processing_time, 3),
                "stages": {
                    "stage1_structural": stage1_length,
                    "stage2_whitespace": stage2_length,
                    "stage3_characters": stage3_length,
                    "stage4_content": final_length
                }
            }
        }
        
        logger.info(f"Cleaning completed: {original_length} -> {final_length} chars ({result['metadata']['reduction_percentage']}% reduction) in {processing_time:.3f}s")
        
        return result
