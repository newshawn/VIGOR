#!/usr/bin/env python3
"""
Script to extract text content from a PDF file
"""
import sys

try:
    import pypdf
    print("Using pypdf library")
    
    def extract_with_pypdf(pdf_path):
        reader = pypdf.PdfReader(pdf_path)
        text = []
        for i, page in enumerate(reader.pages):
            text.append(f"\n{'='*80}\nPage {i+1}\n{'='*80}\n")
            text.append(page.extract_text())
        return '\n'.join(text)
    
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./assets/10475_Verifier_Free_RL_for_LLM (1).pdf"
    content = extract_with_pypdf(pdf_path)
    print(content)
    
except ImportError:
    try:
        import PyPDF2
        print("Using PyPDF2 library")
        
        def extract_with_pypdf2(pdf_path):
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = []
                for i, page in enumerate(reader.pages):
                    text.append(f"\n{'='*80}\nPage {i+1}\n{'='*80}\n")
                    text.append(page.extract_text())
            return '\n'.join(text)
        
        pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./assets/10475_Verifier_Free_RL_for_LLM (1).pdf"
        content = extract_with_pypdf2(pdf_path)
        print(content)
        
    except ImportError:
        try:
            import pdfplumber
            print("Using pdfplumber library")
            
            def extract_with_pdfplumber(pdf_path):
                text = []
                with pdfplumber.open(pdf_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text.append(f"\n{'='*80}\nPage {i+1}\n{'='*80}\n")
                        text.append(page.extract_text())
                return '\n'.join(text)
            
            pdf_path = sys.argv[1] if len(sys.argv) > 1 else "./assets/10475_Verifier_Free_RL_for_LLM (1).pdf"
            content = extract_with_pdfplumber(pdf_path)
            print(content)
            
        except ImportError:
            print("Error: No PDF library found. Please install one of: pypdf, PyPDF2, or pdfplumber")
            print("Run: pip install pypdf")
            sys.exit(1)

