import os
import time
import pandas as pd
from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import openai
from pinecone import Pinecone, ServerlessSpec
import requests

app = Flask(__name__)
app.secret_key = 'd293bf3b050c83c2a3934b040ee31ee5' # Use your secret key

# Load the OpenAI and Pinecone API keys from the environment
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')  # Load Pinecone API key
pc = Pinecone(api_key=pinecone_api_key)  # Initialize Pinecone

# Route for the home page
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_file = request.files.get('file')

        # Check if a file is uploaded
        if not uploaded_file:
            flash("No file uploaded.", "error")
            return redirect(url_for('upload_file'))

        # Check if the file is a PDF
        if not uploaded_file.filename.endswith('.pdf'):
            flash("Invalid file type. Please upload a PDF.", "error")
            return redirect(url_for('upload_file'))

        file_path = 'temp.pdf'
        
        try:
            # Save the uploaded file
            uploaded_file.save(file_path)

            # Extract text from the PDF
            extracted_text = extract_text_from_pdf(file_path)
            if extracted_text is None:
                flash("Failed to extract text from the PDF.", "error")
                return redirect(url_for('upload_file'))

            # Chunk the extracted text
            chunks = chunking(extracted_text)

            # Generate embeddings for the chunks
            embeddings_df = generate_embeddings(chunks)

            # Insert embeddings into Pinecone
            pinecone_db_insert(embeddings_df)

            # Cleanup: remove the temporary file
            if os.path.exists(file_path):
                os.remove(file_path)

            if embeddings_df is not None:
                # Select only the first 3 rows of the DataFrame
                first_three_rows = embeddings_df.head(3)
                return render_template('output.html', tables=[first_three_rows.to_html(classes='data')], titles=first_three_rows.columns.values)
            else:
                flash("Failed to generate embeddings.", "error")
                return redirect(url_for('upload_file'))

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(url_for('upload_file'))

    return render_template('upload.html')


# Route for the query page
@app.route('/query', methods=['GET', 'POST'])
def query_page():
    if request.method == 'POST':
        user_query = request.form['query']

        if user_query:
            # Use the combined function to handle the query and GPT refinement
            refined_answer = query_and_refine(user_query)
            return render_template('query_result.html', query=user_query, answer=refined_answer)

    return render_template('query.html')

# Route to test the query_and_refine function via form
@app.route('/test-query', methods=['GET', 'POST'])
def test_query():
    if request.method == 'POST':
        user_query = request.form['query']

        if user_query:
            refined_answer = query_and_refine(user_query)
            return render_template('query_test.html', answer=refined_answer)

    return render_template('query_test.html')

# API route for testing the query_and_refine function
@app.route('/api/test-query', methods=['POST'])
def api_test_query():
    data = request.get_json()

    if 'query' not in data:
        return jsonify({"error": "No query provided"}), 400

    user_query = data['query']
    refined_answer = query_and_refine(user_query)

    return jsonify({"query": user_query, "refined_answer": refined_answer}), 200

# Utility function to extract text from a PDF
def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load()  # Loads the pages as LangChain documents
    text = " ".join([page.page_content for page in pages])  # Concatenate text from all pages
    return text

# Utility function to split text into chunks
def chunking(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=1350,
        chunk_overlap=350,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def generate_embeddings(chunks):
    embedding_values = []
    embeddings = []

    for chunk in chunks:
        try:
            embed = openai.Embedding.create(
                input=chunk,
                model="text-embedding-ada-002"
            )
            time.sleep(1)  # Pause to respect API rate limits

            embeddings.append(embed)
            specific_values = embed['data'][0]['embedding']
            embedding_values.append(specific_values)

        except Exception as e:
            print(f"Error generating embedding for chunk: {chunk[:30]}... | Error: {str(e)}")
            # You can choose to handle the error further (e.g., skip the chunk or return None)

    # Create DataFrame only if we have valid embeddings
    if embedding_values:
        data = {
            'chunk': chunks[:len(embedding_values)],  # Align with the number of successful embeddings
            'embedding_values': embedding_values
        }
        df = pd.DataFrame(data)
        return df
    else:
        print("No valid embeddings were generated.")
        return None  # Return None if no embeddings were successful


# Function to insert embeddings into Pinecone
def pinecone_db_insert(df):
    index_name = 'slck'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=len(df['embedding_values'][0]),
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

    index = pc.Index(index_name)
    
    # Create a list of vectors with metadata
    vectors = [
        (str(i), df['embedding_values'][i], {"chunk": df['chunk'][i]}) for i in range(len(df))
    ]
    
    # Upsert vectors into Pinecone
    index.upsert(vectors=vectors)

def post_to_slack(webhook_url, message):
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "text": message
    }
    response = requests.post(webhook_url, headers=headers, json=data)

    if response.status_code == 200:
        print("Message posted successfully.")
    else:
        print(f"Failed to post message. Status code: {response.status_code}, Response: {response.text}")

# Your Slack Incoming Webhook URL
webhook_url = 'https://hooks.slack.com/services/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'

def query_and_refine(user_query):
    embeddings = openai.Embedding.create(
        input=user_query,
        model="text-embedding-ada-002"
    )

    query_result = embeddings['data'][0]['embedding']
    index_name = 'slck'

    index = pc.Index(index_name)
    answer = index.query(
        vector=list(query_result),
        top_k=1,
        include_metadata=True
    )

    if len(answer['matches']) > 0:
        ans_text = answer['matches'][0].get('metadata', {}).get('chunk', "No chunk found.")
    else:
        return "No relevant data found."

    gpt_response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "system",
                "content": f"Answer the query based strictly on the following data. Do not add anything extra:\n\n{ans_text}"
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    refined_answer = gpt_response['choices'][0]['message']['content']

    # Post message to Slack
    post_to_slack(webhook_url, refined_answer)

    return refined_answer



if __name__ == '__main__':
    app.run(debug=True)
