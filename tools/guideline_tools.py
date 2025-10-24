"""
Guideline retrieval tools for the Agentic RAG Medical Documentation System.
Handles hospital guideline search and information extraction.
"""

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from core.database import db_manager
from config.llm_config import llm_config
from config.settings import SIMILARITY_SCORE_THRESHOLD, GUIDELINE_SEARCH_K

def retrieve_guidelines_by_section(query: str) -> dict:
    """
    First finds the most relevant document in the vector database, then retrieves all its sections while skipping 'Unknown Section'.
    Used as a helper function for 'start_document_generation' gathering all sections of the guideline before proceeding to generate it section by section.
    
    Args:
        query (str): The query to search for relevant guidelines.
    
    Returns:
        dict: Dictionary of guideline sections with section titles as keys and their content as values.
        Returns an error message string if no guidelines are found.
    """
    # Find the most relevant document using configurable k value
    print(f"[INFO] Searching for the relevant guideline.")
    doc_results = db_manager.guideline_db.similarity_search(query, k=GUIDELINE_SEARCH_K)
    if not doc_results:
        return "No relevant guidelines found."
    
    best_doc = doc_results[0].metadata["document_name"]  # Identify the most relevant document
    print(f"[DEBUG] Identified most relevant guideline document: {best_doc}")
    
    # Retrieve all sections for the identified document
    section_results = db_manager.guideline_db.get(where={"document_name": best_doc})
    
    if not section_results["documents"]:
        return "No relevant sections found in the selected guideline document."
    
    # Organize sections by title while skipping 'Unknown Section'
    # Unknown sections are defined in the vectordatabase logic as sections without a headline
    print(f"[INFO] Fetching all sections from guideline: {best_doc}")
    sections = {}
    for chunk_text, metadata in zip(section_results["documents"], section_results["metadatas"]):
        section_title = metadata.get("section_title", "Unknown Section")
        
        if section_title == "Unknown Section":
            print(f"[WARNING] Skipping 'Unknown Section' in {best_doc}.")
            continue  # Skip processing this section
        
        if section_title not in sections:
            sections[section_title] = ""
        sections[section_title] += chunk_text + "\n"

    # **DEBUG OUTPUT: Verify the final structure**
    print(f"[DEBUG] Retrieved Sections: {sections.keys()}")
    print("[RESULT] All sections has been gathered.")
    return sections

@tool
def retrieve_guideline_knowledge(query: str) -> str:
    """
    Retrieve relevant sections from official hospital guidelines based on a natural language query.

    This tool performs semantic search with similarity scoring to identify and rank relevant content
    from vectorized hospital guideline documents. Low-relevance results are filtered out using a
    similarity threshold. Matching content is grouped by section and document origin, and further
    filtered when specific terms (e.g., section titles or document names) appear in the query.
    The final result includes structured, context-aware content along with a listing of available
    sections for further exploration.

    Args:
        query (str): A natural language query related to clinical guidelines or procedures.

    Returns:
        str: A Markdown-formatted summary of guideline content, filtered by semantic relevance
            and organized by section and document for clinical reference.
    """
    # Perform similarity search with scores using configurable k values
    paragraph_raw = db_manager.guideline_db.similarity_search_with_score(
        query, k=GUIDELINE_SEARCH_K, filter={"granularity": "paragraph"}
    )
    section_raw = db_manager.guideline_db.similarity_search_with_score(
        query, k=GUIDELINE_SEARCH_K, filter={"granularity": "section"}
    )

    # Filter out low-scoring matches
    paragraph_results = [doc for doc, score in paragraph_raw if score >= SIMILARITY_SCORE_THRESHOLD]
    section_results = [doc for doc, score in section_raw if score >= SIMILARITY_SCORE_THRESHOLD]
    
    # Merge all 
    results = []
    for res in paragraph_results + section_results:
        results.append(res)

    if not results:
        return "No relevant guideline found for this query."
    
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
        return "No relevant guideline sections matched your specific query criteria."
    
    # Sort chunks within each section by their index to maintain document order
    for key in grouped:
        grouped[key]["chunks"].sort(key=lambda x: x["index"])
        grouped[key]["content"] = [chunk["content"] for chunk in grouped[key]["chunks"]]
    
    # Build formatted output
    output_lines = ["# Guideline Search Results\n"]
    
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
        output_lines.append(f"*From guideline: {doc_name}*\n")
        
        # Join content with paragraph breaks for readability
        section_content = "\n\n".join(doc_info["content"])
        output_lines.append(section_content)
        output_lines.append("\n---\n")
    
    # Add section listing for each document found
    output_lines.append("\n# Available Sections by Document\n")
    output_lines.append("The following sections are available in the guidelines matched by your query:\n")
    
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