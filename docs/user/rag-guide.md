# RAG Knowledge Base User Guide

This guide covers how to use the Retrieval-Augmented Generation (RAG) features in ATLAS to enhance AI responses with your own documents and knowledge.

## Overview

RAG (Retrieval-Augmented Generation) allows ATLAS to search through your documents and include relevant context when generating responses. This helps the AI provide more accurate, contextual answers based on your specific documentation, code, or knowledge base.

## Getting Started

### Opening the Knowledge Base Manager

1. Click the **Knowledge Bases** option in the sidebar navigation
2. The KB Manager window will open, showing your existing knowledge bases

### Creating a Knowledge Base

1. In the KB Manager, click the **+** (Add) button next to "Knowledge Bases"
2. Enter a name for your knowledge base (e.g., "Project Documentation")
3. Add an optional description
4. Select an embedding model:
   - **all-MiniLM-L6-v2**: Fast, local model (default)
   - **text-embedding-3-small**: OpenAI's compact embedding model
   - **text-embedding-ada-002**: OpenAI's legacy model
   - **embed-english-v3.0**: Cohere's English model
5. Click **Create**

## Uploading Documents

### File Upload

1. Select a knowledge base in the KB Manager
2. Click **Upload Documents**
3. Choose how to add files:
   - **Browse Files**: Select individual files
   - **Browse Folder**: Add all files from a folder recursively
   - Drag and drop files directly into the dialog

### Supported File Types

| Category | Extensions |
| -------- | ---------- |
| Text | .txt, .md, .markdown |
| Documents | .pdf, .html, .htm |
| Data | .json, .csv, .xml |
| Code | .py, .js, .ts, .java, .go, .rs, and more |
| Config | .yaml, .yml, .toml |

### URL Ingestion

1. In the upload dialog, switch to the **URL** tab
2. Enter a web page URL
3. Click **Add** to queue the URL
4. Click **Upload** to fetch and ingest the content

### Metadata Options

Before uploading, you can add optional metadata:

- **Tags**: Comma-separated labels for organization
- **Source**: Reference URL or identifier for provenance
- **Check for duplicates**: Enable to skip files with identical content already in the KB

## Managing Knowledge Bases

### Browsing Documents

1. Select a knowledge base to see its documents
2. Click a document to view its chunks
3. Use the search box to filter documents

### Viewing Chunks

Chunks are the segments your documents are split into for retrieval. Each chunk:

- Has a preview showing its content
- Displays its position index
- Can be edited if needed

### Editing Chunks

1. Select a chunk from the list
2. Toggle the **Edit Mode** switch
3. Modify the content in the editor
4. Click **Save Changes** or **Revert** to undo

### Embedding Visualization

The **Embeddings** tab provides a 2D visualization of your chunks:

1. Switch to the Embeddings tab
2. Select a reduction method (t-SNE or PCA)
3. Explore the scatter plot:
   - Points are color-coded by document
   - Hover to see chunk previews
   - Click to select a chunk

This helps you understand the semantic relationships between your content.

### Query Testing

Test how RAG retrieval works with your content:

1. Switch to the **Query Test** tab
2. Enter a test query
3. Adjust the number of results (1-20)
4. Click **Search**
5. Review the retrieved chunks with similarity scores

Use this to verify your content is being retrieved correctly for expected queries.

## Configuration

### Per-KB Settings

Click the ⚙️ (Configure) button to adjust settings:

- **Name**: Update the knowledge base name
- **Description**: Update the description
- **Chunk Size**: Target token count for chunks (100-2000)
- **Chunk Overlap**: Token overlap between adjacent chunks (0-500)

These settings affect how new documents are processed.

## Export and Import

### Exporting a Knowledge Base

1. Select the knowledge base
2. Click **Export**
3. Choose a save location
4. A ZIP file is created with all documents and metadata

### Importing a Knowledge Base

1. Click **Import** in the KB Manager header
2. Select a previously exported ZIP file
3. The KB is restored with "(Imported)" suffix
4. Documents are re-indexed with the current embedding model

## Using RAG in Conversations

Once you have documents in your knowledge base:

1. Start or continue a conversation
2. Enable RAG by selecting a knowledge base in the chat settings
3. Ask questions related to your content
4. ATLAS will automatically retrieve relevant context and include it in responses

## Best Practices

### Document Organization

- Create separate knowledge bases for different topics or projects
- Use descriptive titles for documents
- Add tags to help with organization
- Keep documents focused on single topics when possible

### Chunk Size Optimization

- **Smaller chunks (200-300)**: Better for precise retrieval, more granular
- **Larger chunks (500-1000)**: More context per result, fewer API calls
- **Overlap (50-100)**: Helps maintain context across chunk boundaries

### Content Quality

- Remove unnecessary formatting and boilerplate
- Ensure documents are complete and self-contained
- Update documents when source material changes
- Use duplicate detection to avoid redundant content

## Troubleshooting

### No Results Found

- Check that documents are fully indexed (not "pending")
- Verify the query relates to your content
- Try increasing the number of results
- Ensure the knowledge base is selected for the conversation

### Poor Result Quality

- Consider adjusting chunk size
- Review and edit chunk content if needed
- Check if the embedding model is appropriate for your content
- Add more relevant documents

### Slow Performance

- Large knowledge bases may take longer to search
- Consider using a local embedding model for faster processing
- Reduce the number of results requested

## Related Documentation

- [Configuration Reference](../configuration.md) - RAG settings
- [Architecture Overview](../architecture-overview.md) - RAG system design
- [Developer Guide](../developer/rag-integration.md) - API integration
