# sample JSON output
- DO NOT OUTPUT ANY TEXT OUTSIDE THE JSON STRUCTURE.
- Example output:
- DO NOT OUTPUT ANY OTHER TEXT. JUST RETURN VALID JSON.

# the following Claude 3 Haiku prompt to generate context for each chunk from Anthropic. The resulting contextual text, usually 50-100 tokens, is prepended to the chunk before embedding it and before creating the BM25 index.
- Claude 3 Haiku: 200,000 tokens, $0.002/1k tokens
- OpenAI: 125,000 tokens, $0.0015/1k tokens
- Gemini: 300,000 tokens, $0.001/1k tokens
- Llama3.2: 128,000 tokens, $0.002/1k tokens

<document> 
{{WHOLE_DOCUMENT}} 
</document> 
Here is the chunk we want to situate within the whole document 
<chunk> 
{{CHUNK_CONTENT}} 
</chunk> 
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 

