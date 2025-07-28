"""
Output Writer for PDF Outline Extraction
Adobe Hackathon "Connecting the Dots" Challenge - Round 1A

Handles JSON output generation with validation and error handling.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import io



class OutputWriter:
    """
    Handles writing structured PDF outline data to JSON files.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def write_outline(self, outline_data: Dict[str, Any], output_path: Path) -> bool:
        """
        Write outline data to a JSON file.
        
        Args:
            outline_data: Dictionary containing title and outline
            output_path: Path to the output JSON file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate input data
            validated_data = self._validate_outline_data(outline_data)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(
                    validated_data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=False
                )
            
            self.logger.info(f"Successfully wrote outline to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write outline to {output_path}: {str(e)}")
            return False
    
    def _validate_outline_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and clean outline data before writing.
        
        Args:
            data: Raw outline data
            
        Returns:
            Validated and cleaned data
            
        Raises:
            ValueError: If data structure is invalid
        """
        if not isinstance(data, dict):
            raise ValueError("Outline data must be a dictionary")
        
        # Ensure required fields exist
        title = data.get('title', 'Untitled Document')
        outline = data.get('outline', [])
        
        # Validate title
        if not isinstance(title, str):
            title = str(title) if title else 'Untitled Document'
        
        # Clean title
        title = self._clean_text(title)
        if not title:
            title = 'Untitled Document'
        
        # Validate outline structure
        validated_outline = self._validate_outline_entries(outline)
        
        return {
            'title': title,
            'outline': validated_outline
        }
    
    def _validate_outline_entries(self, outline: List[Any]) -> List[Dict[str, Any]]:
        """
        Validate and clean outline entries.
        
        Args:
            outline: List of outline entries
            
        Returns:
            List of validated outline entries
        """
        if not isinstance(outline, list):
            self.logger.warning("Outline is not a list, converting to empty list")
            return []
        
        validated_entries = []
        
        for i, entry in enumerate(outline):
            try:
                validated_entry = self._validate_single_entry(entry, i)
                if validated_entry:
                    validated_entries.append(validated_entry)
            except Exception as e:
                self.logger.warning(f"Skipping invalid outline entry {i}: {str(e)}")
                continue
        
        return validated_entries
    
    def _validate_single_entry(self, entry: Any, index: int) -> Optional[Dict[str, Any]]:
        """
        Validate a single outline entry.
        
        Args:
            entry: Single outline entry
            index: Entry index for error reporting
            
        Returns:
            Validated entry or None if invalid
            
        Raises:
            ValueError: If entry is fundamentally invalid
        """
        if not isinstance(entry, dict):
            raise ValueError(f"Entry {index} is not a dictionary")
        
        # Extract and validate required fields
        level = entry.get('level', '')
        text = entry.get('text', '')
        page = entry.get('page', 1)
        
        # Validate level
        if level not in ['H1', 'H2', 'H3']:
            # Try to infer or fix level
            if isinstance(level, str) and level.lower().startswith('h'):
                level_num = level[1:] if len(level) > 1 else '1'
                if level_num in ['1', '2', '3']:
                    level = f'H{level_num}'
                else:
                    level = 'H1'  # Default to H1
            else:
                level = 'H1'  # Default to H1
        
        # Validate text
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Entry {index} has invalid or empty text")
        
        text = self._clean_text(text)
        if not text:
            raise ValueError(f"Entry {index} has empty text after cleaning")
        
        # Validate page number
        try:
            page = int(page)
            if page < 1:
                page = 1
        except (ValueError, TypeError):
            page = 1
        
        return {
            'level': level,
            'text': text,
            'page': page
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Limit length
        if len(text) > 500:
            text = text[:497] + '...'
        
        return text
    
    def write_batch_summary(self, results: List[Dict[str, Any]], output_dir: Path) -> bool:
        """
        Write a summary of batch processing results.
        
        Args:
            results: List of processing results
            output_dir: Output directory
            
        Returns:
            True if successful, False otherwise
        """
        try:
            summary_data = {
                'processing_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'total_files': len(results),
                    'successful': sum(1 for r in results if r.get('success', False)),
                    'failed': sum(1 for r in results if not r.get('success', False)),
                    'files': results
                }
            }
            
            summary_path = output_dir / 'processing_summary.json'
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Wrote processing summary to {summary_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write processing summary: {str(e)}")
            return False
    
    def validate_json_output(self, json_path: Path) -> bool:
        """
        Validate that a JSON output file meets the requirements.
        
        Args:
            json_path: Path to JSON file to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check required structure
            if not isinstance(data, dict):
                self.logger.error(f"JSON file {json_path} is not a dictionary")
                return False
            
            if 'title' not in data or 'outline' not in data:
                self.logger.error(f"JSON file {json_path} missing required fields")
                return False
            
            # Validate title
            if not isinstance(data['title'], str):
                self.logger.error(f"Title in {json_path} is not a string")
                return False
            
            # Validate outline
            outline = data['outline']
            if not isinstance(outline, list):
                self.logger.error(f"Outline in {json_path} is not a list")
                return False
            
            # Validate each outline entry
            for i, entry in enumerate(outline):
                if not isinstance(entry, dict):
                    self.logger.error(f"Outline entry {i} in {json_path} is not a dictionary")
                    return False
                
                required_fields = ['level', 'text', 'page']
                for field in required_fields:
                    if field not in entry:
                        self.logger.error(f"Outline entry {i} in {json_path} missing field: {field}")
                        return False
                
                # Validate field types and values
                if entry['level'] not in ['H1', 'H2', 'H3']:
                    self.logger.error(f"Invalid level '{entry['level']}' in entry {i} of {json_path}")
                    return False
                
                if not isinstance(entry['text'], str) or not entry['text'].strip():
                    self.logger.error(f"Invalid text in entry {i} of {json_path}")
                    return False
                
                if not isinstance(entry['page'], int) or entry['page'] < 1:
                    self.logger.error(f"Invalid page number in entry {i} of {json_path}")
                    return False
            
            self.logger.info(f"JSON file {json_path} is valid")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate JSON file {json_path}: {str(e)}")
            return False
    
    def create_empty_outline(self, output_path: Path, title: str = "Empty Document") -> bool:
        """
        Create an empty outline file for documents with no detectable structure.
        
        Args:
            output_path: Path to output file
            title: Document title
            
        Returns:
            True if successful, False otherwise
        """
        empty_data = {
            'title': title,
            'outline': []
        }
        
        return self.write_outline(empty_data, output_path) 