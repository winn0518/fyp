# app.py - Flask Backend
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import shutil
import tempfile
import time
import json
import gc
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Document loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_ibm import WatsonxLLM

# Load environment variables
load_dotenv()

# Configuration from .env
IBM_URL = os.getenv("IBM_URL", "https://us-south.ml.cloud.ibm.com")
IBM_API_KEY = os.getenv("IBM_API_KEY")
IBM_PROJECT_ID = os.getenv("IBM_PROJECT_ID")

# Flask app setup
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize components
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
persist_directory = ".chromadb"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_ibm_credentials():
    """Verify IBM credentials are available"""
    if not IBM_API_KEY or not IBM_PROJECT_ID:
        return False, "IBM credentials not found. Please add IBM_API_KEY and IBM_PROJECT_ID to your .env file"
    return True, ""

# Function to clear database
def clear_database():
    """Delete the existing vector database with proper cleanup"""
    try:
        if os.path.exists(persist_directory):
            # Force close any open vectorstore connections
            try:
                gc.collect()
            except:
                pass
            
            # Add delay to ensure all file handles are released
            time.sleep(1)
            
            # Strategy 1: Try normal shutil.rmtree first
            try:
                shutil.rmtree(persist_directory)
                return True
            except (PermissionError, OSError):
                pass
            
            # Strategy 2: If normal deletion fails, try removing files recursively
            try:
                def remove_tree(path):
                    """Recursively remove all files and directories"""
                    if os.path.isdir(path):
                        for item in os.listdir(path):
                            item_path = os.path.join(path, item)
                            if os.path.isdir(item_path):
                                remove_tree(item_path)
                            else:
                                try:
                                    os.remove(item_path)
                                except:
                                    import stat
                                    os.chmod(item_path, stat.S_IWRITE)
                                    os.remove(item_path)
                        os.rmdir(path)
                
                remove_tree(persist_directory)
                return True
            except Exception as e:
                print(f"Error clearing database: {e}")
                return False
    except Exception as e:
        print(f"Error clearing database: {e}")
    return False

# Function to load documents based on file type
def load_document(file_path, file_type):
    """Load document based on file type"""
    try:
        if file_type == 'pdf':
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_type == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
        elif file_type == 'docx':
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        else:
            return []
        
        return documents
    except Exception as e:
        print(f"Error loading {file_type} file: {str(e)}")
        return []

# Function to process documents
def process_documents(file_paths):
    """Process uploaded documents"""
    all_docs = []
    
    try:
        for file_path, original_filename in file_paths:
            # Determine file type
            if original_filename.lower().endswith('.pdf'):
                file_type = 'pdf'
            elif original_filename.lower().endswith('.txt'):
                file_type = 'txt'
            elif original_filename.lower().endswith('.docx'):
                file_type = 'docx'
            else:
                continue
            
            documents = load_document(file_path, file_type)
            
            if documents:
                # Add metadata
                for doc in documents:
                    doc.metadata["source"] = original_filename
                    doc.metadata["original_filename"] = original_filename
                    if file_type == 'pdf' and 'page' in doc.metadata:
                        doc.metadata["page_number"] = doc.metadata["page"] + 1
                
                all_docs.extend(documents)
        
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
    
    return all_docs

# Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('webpage.html')

@app.route('/api/check-database', methods=['GET'])
def check_database():
    """Check if database exists"""
    db_exists = os.path.exists(persist_directory)
    return jsonify({
        'database_exists': db_exists,
        'status': 'success'
    })

@app.route('/api/clear-database', methods=['POST'])
def api_clear_database():
    """Clear the vector database"""
    try:
        success = clear_database()
        if success:
            return jsonify({
                'status': 'success',
                'message': 'Database cleared successfully'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to clear database'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/upload-documents', methods=['POST'])
def upload_documents():
    """Handle document upload and processing"""
    # Check credentials
    creds_ok, creds_msg = check_ibm_credentials()
    if not creds_ok:
        return jsonify({
            'status': 'error',
            'message': creds_msg
        }), 400
    
    # Check if files were uploaded
    if 'files' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No files uploaded'
        }), 400
    
    files = request.files.getlist('files')
    if len(files) == 0 or files[0].filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No files selected'
        }), 400
    
    # Get processing parameters
    chunk_size = request.form.get('chunk_size', 1000, type=int)
    chunk_overlap = request.form.get('chunk_overlap', 200, type=int)
    
    # Clear existing database
    if os.path.exists(persist_directory):
        clear_database()
    
    file_paths = []
    processed_files = []
    
    try:
        # Save uploaded files
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                file_paths.append((file_path, filename))
                processed_files.append(filename)
        
        if not file_paths:
            return jsonify({
                'status': 'error',
                'message': 'No valid files uploaded'
            }), 400
        
        # Process documents
        documents = process_documents(file_paths)
        
        if not documents:
            return jsonify({
                'status': 'error',
                'message': 'Failed to process documents'
            }), 500
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents(documents)
        
        # Create and persist vector database
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectorstore.persist()
        
        # Clean up uploaded files
        for file_path, _ in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully processed {len(processed_files)} documents',
            'data': {
                'files_processed': len(processed_files),
                'total_chunks': len(splits),
                'total_pages': len(documents),
                'avg_chunk_size': sum(len(d.page_content) for d in splits)//len(splits) if splits else 0
            }
        })
        
    except Exception as e:
        # Clean up any remaining files
        for file_path, _ in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        return jsonify({
            'status': 'error',
            'message': f'Error processing documents: {str(e)}'
        }), 500

@app.route('/api/ask-question', methods=['POST'])
def ask_question():
    """Handle question answering"""
    # Check credentials
    creds_ok, creds_msg = check_ibm_credentials()
    if not creds_ok:
        return jsonify({
            'status': 'error',
            'message': creds_msg
        }), 400
    
    # Check if database exists
    if not os.path.exists(persist_directory):
        return jsonify({
            'status': 'error',
            'message': 'Please upload and process documents first'
        }), 400
    
    # Get request data
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No question provided'
        }), 400
    
    question = data['question'].strip()
    k_value = data.get('k_value', 4)
    temperature = data.get('temperature', 0.1)
    
    if not question:
        return jsonify({
            'status': 'error',
            'message': 'Please enter a question'
        }), 400
    
    try:
        # Load existing vectorstore
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Initialize IBM Granite
        llm = WatsonxLLM(
            model_id="ibm/granite-3-3-8b-instruct",
            url=IBM_URL,
            apikey=IBM_API_KEY,
            project_id=IBM_PROJECT_ID,
            params={
                "temperature": temperature,
                "max_new_tokens": 512,
                "repetition_penalty": 1.1,
                "top_p": 0.9
            }
        )
        
        # Create prompt template
        prompt_template = """You are an intelligent document assistant. Use the following context from uploaded documents to answer the question. 
        If the answer cannot be found in the context, say "I cannot find this information in the provided documents."
        Provide clear citations to the source documents when possible.

        Context Information:
        {context}

        Question: {question}

        Please provide a detailed answer based on the context:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Setup RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": k_value}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        # Get answer
        start_time = time.time()
        result = qa_chain({"query": question})
        response_time = time.time() - start_time
        
        # Format sources
        sources = []
        for i, doc in enumerate(result["source_documents"], 1):
            sources.append({
                'id': i,
                'source': doc.metadata.get('source', 'Unknown Document'),
                'page': doc.metadata.get('page_number', doc.metadata.get('page', 'N/A')),
                'content': doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content
            })
        
        return jsonify({
            'status': 'success',
            'data': {
                'answer': result["result"],
                'sources': sources,
                'metrics': {
                    'response_time': round(response_time, 2),
                    'sources_used': len(result["source_documents"]),
                    'confidence': min(len(result["source_documents"]) / k_value * 100, 100)
                }
            }
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing question: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Move webpage.html to templates directory
    if os.path.exists('webpage.html'):
        shutil.move('webpage.html', 'templates/webpage.html')
    
    app.run(debug=True, port=5000)