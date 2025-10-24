"""
Document generation and retrieval tools for the Agentic RAG Medical Documentation System.
Handles generated document operations and document creation initiation.
"""

from langchain_core.tools import tool

from core.database import db_manager
from core.memory import memory_manager
from tools.guideline_tools import retrieve_guidelines_by_section
from config.settings import SIMILARITY_SCORE_THRESHOLD, GENERATED_DOC_SEARCH_K

@tool
def start_document_generation(query: str) -> str:
    """
    Initiates the generation of a new document by retrieving relevant guideline sections based on the input query.

    This tool searches for the most relevant guideline document, extracts its structured content (excluding sections
    titled 'Unknown Section'), and stores the result in the memory manager for future reference.

    Args:
        query (str): A user-provided prompt or question that describes what kind of document is needed 
                     (e.g., "Skriv en plejeforlÃ¸bsplan for denne patient").

    Returns:
        str: A confirmation message indicating that the document has been created and is available for further queries.
    """
    guidelines = retrieve_guidelines_by_section(query)
    memory_manager.retrieved_guidelines = guidelines
    return """Det Ã¸nskede dokument er blevet oprettet og er nu tilgÃ¦ngeligt for sÃ¸gning med vÃ¦rktÃ¸jet 'retrieve_generated_document_info'.
            FortÃ¦l brugeren, at dokumentet er blevet oprettet og gemt, og at du nu er klar til at besvare yderligere spÃ¸rgsmÃ¥l om dokumentet."""

@tool
def retrieve_generated_document_info(query: str) -> str:
    """
    Retrieve relevant content from previously generated medical documents based on a user query.

    This tool performs semantic similarity search across vectorized final documents, filters out
    low-relevance matches using a similarity score threshold, and groups high-quality results
    by section and document origin. If specific sections or document names are mentioned in the
    query, the output is filtered accordingly. The final result presents a structured overview
    of matched content and a listing of available sections for further exploration.

    Args:
        query (str): A natural language query concerning information in the generated medical document.

    Returns:
        str: A structured Markdown-formatted overview of relevant document content,
            filtered by semantic score and organized for clinical interpretation.
    """
    # Check if final document exists
    try:
        doc_count = db_manager.generated_docs_db._collection.count()
        if doc_count == 0:
            return "No final document has been generated yet. Please use the 'start_document_generation' tool to create a medical document first."
    except:
        return "No final document has been generated yet. Please use the 'start_document_generation' tool to create a medical document first."
    
    # Perform similarity search with scores using configurable k values
    paragraph_raw = db_manager.generated_docs_db.similarity_search_with_score(
        query, k=GENERATED_DOC_SEARCH_K, filter={"granularity": "paragraph"}
    )
    section_raw = db_manager.generated_docs_db.similarity_search_with_score(
        query, k=GENERATED_DOC_SEARCH_K, filter={"granularity": "section"}
    )
    summary_raw = db_manager.generated_docs_db.similarity_search_with_score(
        query, k=1, filter={"section_title": "Document Summary"}
    )

    # Filter out low-scoring matches
    paragraph_results = [doc for doc, score in paragraph_raw if score >= SIMILARITY_SCORE_THRESHOLD]
    section_results = [doc for doc, score in section_raw if score >= SIMILARITY_SCORE_THRESHOLD]
    summary_results = [doc for doc, score in summary_raw if score >= SIMILARITY_SCORE_THRESHOLD]
    
    # Merge all 
    results = []
    for res in paragraph_results + section_results + summary_results:
            results.append(res)

    if not results:
        return "No relevant document found for this query."

    # Try to identify specific sections or documents mentioned in the query
    query_lower = query.lower()
    
    # Extract all unique sections and document names for analysis
    all_sections = set()
    all_documents = set()
    
    # Track all sections by document for the complete listing
    document_sections = {}

    for doc in results: # extract metadata for later filtering and display
        section = doc.metadata.get("section_title", "").lower()
        doc_name = doc.metadata.get("document_name", "").lower()
        doc_id = doc.metadata.get("document_id", "")
        
        if section:
            all_sections.add(section)
        if doc_name:
            all_documents.add(doc_name)
        
        # Build a mapping of document_id -> [section_titles]
        if doc_id not in document_sections:
            document_sections[doc_id] = {
                "name": doc.metadata.get("document_name", "Unknown Document"),
                "sections": set()
            }
        if section:
            document_sections[doc_id]["sections"].add(section)
    
    # Check if query contains any specific section or document references
    likely_sections = [section for section in all_sections if section in query_lower]
    likely_documents = [doc_name for doc_name in all_documents if doc_name.replace(".pdf", "") in query_lower]

    # Group results by document and section
    grouped = {}
    for doc in results:
        doc_name = doc.metadata.get("document_name", "Unknown Document")
        section = doc.metadata.get("section_title", "Unknown Section").strip()
        doc_id = doc.metadata.get("document_id", "")
        chunk_index = doc.metadata.get("chunk_index", 0)
        
        # Apply filters if specific section or document was detected
        if (likely_sections and section.lower() not in likely_sections) or (likely_documents and doc_name.lower() not in likely_documents):
            continue
            
        key = f"{doc_name} | {section}"
        if key not in grouped:
            grouped[key] = {
                "chunks": [],
                "doc_id": doc_id,
                "indices": [],
                "content": []
            }
        
        grouped[key]["chunks"].append({
            "index": chunk_index,
            "content": doc.page_content
        })
        grouped[key]["indices"].append(chunk_index)
        grouped[key]["content"].append(doc.page_content)
    
    if not grouped:
        return "No relevant document sections matched your specific query criteria."
    
    # Sort chunks within each section by their index to maintain document order
    for key in grouped:
        grouped[key]["chunks"].sort(key=lambda x: x["index"])
        grouped[key]["content"] = [chunk["content"] for chunk in grouped[key]["chunks"]]
    
    # Build formatted output
    output_lines = ["# Generated Document Search Results\n"]
    
    # Add filter information if applicable
    filters_applied = []

    if likely_sections:
        filters_applied.append(f"Sections: {', '.join(likely_sections)}")
    if likely_documents:
        filters_applied.append(f"Documents: {', '.join([d.replace('.pdf', '') for d in likely_documents])}")

    # Sort groups by document name for consistent output
    sorted_keys = sorted(grouped.keys())
    
    # Track which documents we've processed for the content sections
    processed_doc_ids = set()
    
    for key in sorted_keys:
        doc_info = grouped[key]
        doc_id = doc_info["doc_id"]
        processed_doc_ids.add(doc_id)
        
        # Extract document and section names
        doc_name, section = key.split(" | ")
        
        # Create section header with document reference
        output_lines.append(f"## {section}")
        output_lines.append(f"*From document: {doc_name}*\n")
        
        # Join content with paragraph breaks for readability
        section_content = "\n\n".join(doc_info["content"])
        output_lines.append(section_content)
        output_lines.append("\n---\n")
    
    # Add section listing for each document found
    output_lines.append("\n# Available Sections by Document\n")
    output_lines.append("The following sections are available in the documents matched by your query:\n")
    
    for doc_id, doc_data in document_sections.items():
        # Highlight which sections were included vs. not included in results
        doc_name = doc_data["name"]
        output_lines.append(f"## {doc_name}\n")
        
        if len(doc_data["sections"]) > 0:
            sections_list = sorted(doc_data["sections"])
            for section in sections_list:
                # Check if this section was included in the results
                section_key = f"{doc_name} | {section}"
                if section_key in grouped:
                    output_lines.append(f"+ {section} (included in results)")
                else:
                        output_lines.append(f"? {section} (not included in results)")
        else:
            output_lines.append("- No section information available for this document")
        
        output_lines.append("")
    
    # Add usage tips
    output_lines.append("\n**Tips:**")
    output_lines.append("- For more specific results, mention the section or document name in your query")
    output_lines.append("- To see content from sections marked '?', try rephrasing your query to include those section names")
    output_lines.append("- For full document content, include the document name in your query without specific section terms")
    
    return "\n".join(output_lines)