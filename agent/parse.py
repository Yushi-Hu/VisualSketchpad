class Parser:
    def parse(self, response):
        if isinstance(response, dict) and 'content' in response:
            response = response['content']
        oring_content = response.replace("\_", "_")
        content = oring_content.replace("\\", "")
        
        try:
            
            start_pos = content.find("```python")
            if start_pos != -1:
                content = content[start_pos+len("```python"):]

            end_pos = content.find("```")
            if end_pos != -1:
                content = content[:end_pos]
            
            if start_pos == -1 or end_pos == -1:
                return {'status': False, 'content': content, 'message': 'Program is NOT enclosed in ```python``` properly.', 'error_code': 'unknown'}
            if len(content) > 0:
                compile(content, "prog.py", "exec")
                return {'status': True, 'content': content, 'message': 'Parsing succeeded.', 'error_code': ''}
            else:
                return {'status': False, 'content': content, 'message': "The content is empty, or it failed to parse the content correctly.", 'error_code': 'unknown'}
        except Exception as err:
            return {'status': False, 'content': content, 'message': f"Unexpected {type(err)}: {err}.", 'error_code': 'unknown'}
    
def main():
    parser = Parser()
    
    # testing 1
    program = """Thought: I thought a lot and here is what I am thinking.\nAction:```python\n"""
    program += """def solve():\n"""
    program += """    output0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    print("\n\n-----------------------------------\n\n")
    
    # testing 2
    program = """Thought: I thought a lot and here is what I am thinking.\nAction:```python\n"""
    program += """def solve():\n"""
    program += """    aha\noutput0 = text_generation(prompt="Would you rather have an Apple Watch - or a BABY?")\n"""  
    program += """    output1 = text_summarization(text=output0["text"])\n"""
    program += """    return output1\n""" 
    program += """```"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    print("\n\n-----------------------------------\n\n")
    
    # testing 3
    program = """Thought: I thought a lot and here is what I am thinking.\nAction: No need"""
    print(program)
    results = parser.parse(program)
    print(results)
    
    
if __name__ == '__main__':
    main()