
# Text and PDF Embeddings

This project contains scripts for generating embeddings for text data and querying indexes of PDF documents. The `modify.py` script can handle large batches of texts and generate embeddings for each text. It also provides a function to generate embeddings for a query text. The `pdf.py` script allows the user to select which indexes to query and whether to include resource documents in the response. The user can then enter a query, and the script will return the search response and the source documents if requested.

## File Structure

- `modify.py`: A script for generating embeddings for text data.
- `pdf.py`: A script for querying indexes of PDF documents.
- `pdf_st.py`: A script for querying indexes of PDF documents in a different mode.
- `requirements.txt`: The requirements file for the project.
