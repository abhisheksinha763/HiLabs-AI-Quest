import pandas as pd
import subprocess
import re
import json
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class IntelligentQueryService:
    def __init__(self, csv_path: str, llm_model: str = "llama3.2:1b"):
        self.csv_path = csv_path
        self.llm_model = llm_model
        self.dataset_info = None
        self._initialize()
    
    def _initialize(self):
        """Initialize by extracting dataset information."""
        self.dataset_info = self._extract_dataset_info()
        if self.dataset_info:
            logger.info(f"Initialized with dataset: {len(self.dataset_info['columns'])} columns, {self.dataset_info['total_rows']} rows")
    
    def _extract_dataset_info(self) -> Optional[Dict[str, Any]]:
        """Extract dataset information from CSV file."""
        try:
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV file not found: {self.csv_path}")
                return None
                
            df = pd.read_csv(self.csv_path)
            columns = df.columns.tolist()
            dtypes = df.dtypes.to_dict()
            sample_rows = df.head(2).to_dict('records') if len(df) > 0 else []

            return {
                'columns': columns,
                'dtypes': {k: str(v) for k, v in dtypes.items()},  # Convert to string for JSON serialization
                'sample_rows': sample_rows,
                'total_rows': len(df)
            }
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return None

    def _call_ollama_llm(self, prompt: str) -> Optional[str]:
        """Call Ollama LLM with the given prompt."""
        try:
            escaped_prompt = prompt.replace('"', '\\"').replace('`', '\\`')
            result = subprocess.run(
                ['ollama', 'run', self.llm_model, escaped_prompt],
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                logger.error(f"Error calling LLM: {result.stderr}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("LLM call timed out")
            return None
        except Exception as e:
            logger.error(f"Error calling LLM: {e}")
            return None

    def _extract_python_code(self, llm_response: str) -> Optional[str]:
        """Extract Python code from LLM response."""
        patterns = [
            r'```python\s*\n(.*?)```',
            r'```\s*\n(.*?)```'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, llm_response, re.DOTALL)
            if matches:
                code = matches[0].strip()
                lines = code.split('\n')
                cleaned_lines = []
                for line in lines:
                    if (not line.strip().startswith('import ') and
                        not line.strip().startswith('# df = pd.read_csv') and
                        not line.strip().startswith('df = pd.read_csv')):
                        cleaned_lines.append(line)

                return '\n'.join(cleaned_lines).strip()

        return None

    def _execute_pandas_code(self, code: str) -> Any:
        """Execute pandas code safely."""
        try:
            safe_builtins = {
                'len': len, 'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'int': int, 'float': float, 'str': str,
                'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
                'set': set, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'sorted': sorted, 'print': print
            }

            local_vars = {
                'pd': pd,
                'df': pd.read_csv(self.csv_path),
                'csv_file_path': self.csv_path
            }
            
            try:
                import numpy as np
                local_vars['np'] = np
            except ImportError:
                pass
                
            exec(code, safe_builtins, local_vars)
            
            # Look for result variables
            result_vars = ['result', 'output', 'answer', 'df_result', 'summary']
            for var in result_vars:
                if var in local_vars:
                    return local_vars[var]
            
            if 'df' in local_vars:
                return local_vars['df']

            return "Code executed successfully but no result variable found"

        except Exception as e:
            logger.error(f"Error executing code: {str(e)}")
            return f"Error executing code: {str(e)}"

    def _generate_analysis_prompt(self, user_query: str) -> str:
        """Generate prompt for analysis code generation."""
        if not self.dataset_info:
            return ""
            
        columns = self.dataset_info['columns']
        dtypes = self.dataset_info['dtypes']
        sample_rows = self.dataset_info['sample_rows']
        total_rows = self.dataset_info['total_rows']
        
        dtype_info = []
        for col, dtype in dtypes.items():
            dtype_info.append(f"  {col}: {dtype}")
            
        sample_info = []
        for i, row in enumerate(sample_rows, 1):
            sample_info.append(f"Row {i}: {row}")

        prompt = f"""
You are a data analyst. I have a healthcare provider CSV file with the following information:

DATASET OVERVIEW:
- Total rows: {total_rows}
- Total columns: {len(columns)}

COLUMNS AND DATA TYPES:
{chr(10).join(dtype_info)}

SAMPLE DATA (first 2 rows):
{chr(10).join(sample_info)}

User Query: "{user_query}"

Based on the user's query and the dataset information above, generate Python pandas code to perform the requested analysis.

Requirements:
1. The CSV data is already loaded in a variable called 'df'
2. Store your final result in a variable called 'result'
3. Use only pandas operations and standard Python libraries
4. DO NOT include any import statements (pandas is already imported as 'pd')
5. DO NOT include code to load the CSV file (df is already available)
6. Handle potential missing values or data type issues appropriately
7. Consider the data types when performing operations
8. Provide meaningful variable names
9. Add comments to explain key steps

Please provide ONLY the Python code wrapped in ```python tags. Do not include any explanations outside the code block.

Example format:
```python
# Your pandas code here
result = df.groupby('column_name').sum()
```
"""
        return prompt

    def _generate_interpretation_prompt(self, user_query: str, code_result: Any) -> str:
        """Generate prompt for result interpretation."""
        if not self.dataset_info:
            return ""
            
        columns = self.dataset_info['columns']
        
        prompt = f"""
You are a data analyst providing insights to a user about healthcare provider data.

Original User Query: "{user_query}"
Available Columns: {', '.join(columns)}
Analysis Result: {str(code_result)[:2000]}

Based on the user's original query and the analysis results above, provide a clear, concise answer to the user's question.

Requirements:
1. Directly answer what the user asked
2. Highlight key findings and patterns
3. Use plain language, avoid technical jargon
4. If there are limitations or assumptions, mention them briefly
5. Keep the response focused and actionable
6. Focus on healthcare provider insights (licenses, locations, specialties, etc.)

Provide your interpretation and insights:
"""
        return prompt

    def process_query(self, user_query: str) -> str:
        """Process user query and return intelligent response."""
        if not self.dataset_info:
            return "Error: Dataset not properly initialized. Please check if the CSV file exists."

        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Generate analysis code
        analysis_prompt = self._generate_analysis_prompt(user_query)
        llm_response = self._call_ollama_llm(analysis_prompt)

        if not llm_response:
            return "Error: Failed to get response from LLM. Please check if Ollama is running."

        # Extract Python code
        python_code = self._extract_python_code(llm_response)
        if not python_code:
            return f"Error: No Python code found in LLM response. Raw response: {llm_response[:500]}"

        logger.info(f"Generated code: {python_code[:200]}...")

        # Execute code
        execution_result = self._execute_pandas_code(python_code)
        logger.info(f"Execution result: {str(execution_result)[:200]}...")

        # Generate interpretation
        interpretation_prompt = self._generate_interpretation_prompt(user_query, execution_result)
        interpretation = self._call_ollama_llm(interpretation_prompt)

        if interpretation:
            return interpretation
        else:
            return f"Analysis completed. Result: {str(execution_result)}"

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Get summary of the dataset."""
        if not self.dataset_info:
            return {"error": "Dataset not initialized"}
        
        return {
            "columns": self.dataset_info['columns'],
            "total_rows": self.dataset_info['total_rows'],
            "column_count": len(self.dataset_info['columns'])
        }
