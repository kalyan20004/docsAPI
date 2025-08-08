from flask import Flask, request, jsonify
import io
import logging
from utils.extractor import extract_text_from_pdf, extract_text_from_docx
from utils.chunker import chunk_text
from utils.embedder import generate_embeddings, embed_query
from utils.faiss_index import create_faiss_index, retrieve_top_k_chunks
from utils.llm import call_gemini

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def extract_text_from_file(file_content, filename):
    """Unified function to extract text from PDF, DOCX, or TXT"""
    try:
        if filename.lower().endswith(".pdf"):
            return extract_text_from_pdf(io.BytesIO(file_content))
        elif filename.lower().endswith(".docx"):
            return extract_text_from_docx(io.BytesIO(file_content))
        elif filename.lower().endswith(".txt"):
            return file_content.decode('utf-8')
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        raise

@app.route('/')
def home():
    return jsonify({
        "message": "RAG API is running. Use POST /api/v1/hackrx/run to query.",
        "health_check": "/health"
    })

@app.route('/api/v1/hackrx/run', methods=['POST'])
def hackrx_webhook():
    try:
        # Get query from form-data field
        query = request.form.get("query")
        if not query:
            return jsonify({"error": "Missing query in form-data"}), 400

        # Get list of uploaded files with key 'documents'
        files = request.files.getlist("documents")
        if not files or len(files) == 0:
            return jsonify({"error": "No documents uploaded"}), 400

        logger.info(f"Processing query: {query}")
        logger.info(f"Processing {len(files)} uploaded documents")

        all_chunks = []

        for i, file in enumerate(files):
            filename = file.filename
            file_content = file.read()

            try:
                # Extract text from uploaded file bytes
                text = extract_text_from_file(file_content, filename)
                if not text or not text.strip():
                    logger.warning(f"No text extracted from {filename}")
                    continue
                
                logger.info(f"Extracted {len(text)} characters from {filename}")

                # Chunk text
                chunks = chunk_text(text, source_file=filename)
                logger.info(f"Created {len(chunks)} chunks from {filename}")

                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                return jsonify({"error": f"Failed to process {filename}: {str(e)}"}), 500

        if not all_chunks:
            return jsonify({"error": "No text could be extracted from any documents"}), 400

        logger.info(f"Total chunks created: {len(all_chunks)}")

        # Generate embeddings
        try:
            embeddings = generate_embeddings(all_chunks)
            if embeddings.size == 0:
                return jsonify({"error": "Failed to generate embeddings"}), 500
            
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return jsonify({"error": f"Failed to generate embeddings: {str(e)}"}), 500

        # Build FAISS index
        try:
            index = create_faiss_index(embeddings)
            if index is None:
                return jsonify({"error": "Failed to build FAISS index"}), 500
            
            logger.info("FAISS index created successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            return jsonify({"error": f"Failed to build FAISS index: {str(e)}"}), 500

        # Embed the query
        try:
            query_embedding = embed_query(query)
            logger.info(f"Query embedding shape: {query_embedding.shape}")
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            return jsonify({"error": f"Failed to embed query: {str(e)}"}), 500

        # Retrieve top matching chunks
        try:
            top_chunks = retrieve_top_k_chunks(query_embedding, index, all_chunks, k=5)
            logger.info(f"Retrieved {len(top_chunks)} relevant chunks")
            
            if not top_chunks:
                logger.warning("No relevant chunks found")
                return jsonify({
                    "status": "rejected",
                    "message": "No relevant information found in documents"
                }), 404
                
        except Exception as e:
            logger.error(f"Error retrieving chunks: {str(e)}")
            return jsonify({"error": f"Failed to retrieve relevant chunks: {str(e)}"}), 500

        # Prepare text chunks for Gemini API
        context_chunks = [chunk['text'] for chunk in top_chunks]
        
        # Log the context being sent to Gemini
        logger.info("Sending context to LLM:")
        for i, chunk in enumerate(context_chunks[:2]):  # Log first 2 chunks
            logger.info(f"Chunk {i+1}: {chunk[:200]}...")

        # Get response from Gemini
        try:
            response = call_gemini(query, context_chunks)
            logger.info("Received response from LLM")
            
            # Add metadata and status
            response['metadata'] = {
                'total_documents': len(files),
                'total_chunks': len(all_chunks),
                'retrieved_chunks': len(top_chunks),
                'sources': list(set([chunk['source'] for chunk in top_chunks]))
            }
            response['status'] = "accepted"
            response['message'] = "Relevant information found and processed"
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Error calling Gemini: {str(e)}")
            return jsonify({"error": f"Failed to process with LLM: {str(e)}"}), 500

    except Exception as e:
        logger.error(f"Unexpected error in hackrx_webhook: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "RAG system is running"}), 200

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
