import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from modify import OpenAIEmbeddings
from termcolor import colored

# Function to load vector databases from a given PDF directory
def load_vector_databases(pdf_directory):
    # List all PDF files in the given directory
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    pdf_indexes = []
    embedding = OpenAIEmbeddings()

    # Iterate through each PDF file
    for pdf_file in pdf_files:
        index_name = os.path.splitext(os.path.basename(pdf_file))[0]
        persist_directory = f'db/{index_name}'

        # Create an index for the PDF file if it doesn't exist
        if not os.path.exists(persist_directory):
            print(f"Creating index for {pdf_file}...")
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            vector_db = Chroma.from_documents(documents=pages, embedding=embedding, persist_directory=persist_directory)
            vector_db.persist()

        # Append the created index to the list of indexes
        pdf_indexes.append({
            "name": index_name,
            "directory": persist_directory
        })

    return pdf_indexes

# Function to load a single or combined indexes based on the input
def load_indexes(indexes_to_query, embedding):
    # If there is more than one index, combine them
    if len(indexes_to_query) > 1:
        combined_index_name = '-'.join(sorted([index["name"] for index in indexes_to_query]))
        combined_persist_directory = f'db/{combined_index_name}'

        # Create a combined index if it doesn't exist
        if not os.path.exists(combined_persist_directory):
            print(f"Creating combined index for {', '.join([index['name'] for index in indexes_to_query])}...")
            combined_docs = []
            for index_to_query in indexes_to_query:
                vector_db = Chroma(persist_directory=index_to_query["directory"], embedding_function=embedding)
                loader = PyPDFLoader(os.path.join("pdfs", f"{index_to_query['name']}.pdf"))
                docs = loader.load_and_split()
                combined_docs.extend(docs)
            combined_vector_db = Chroma.from_documents(documents=combined_docs, embedding=embedding, persist_directory=combined_persist_directory)
            combined_vector_db.persist()
        else:
            # Load the combined index if it exists
            print(f"Loading combined index {combined_index_name}...")
            combined_vector_db = Chroma(persist_directory=combined_persist_directory, embedding_function=embedding)
        return combined_vector_db
    else:
        # Load a single index
        print(f"Loading index {indexes_to_query[0]['name']}...")
        vector_db = Chroma(persist_directory=indexes_to_query[0]["directory"], embedding_function=embedding)
        return vector_db

# Function to run a query on the loaded vector database
def run_query(query, vector_db, include_resources=False):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vector_db.as_retriever(), return_source_documents=include_resources)
    response = qa({"query": query})
    return response

# Main function
def main():
    pdf_directory = "pdfs"
    db_directory = "db"
    
    # Check if the pdf_directory exists, if not, create it
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)

    # Check if the db_directory exists, if not, create it
    if not os.path.exists(db_directory):
        os.makedirs(db_directory)

    pdf_indexes = load_vector_databases(pdf_directory)
    embedding = OpenAIEmbeddings()

    # Load combined indexes
    combined_indexes = [f for f in os.listdir(db_directory) if os.path.isdir(os.path.join(db_directory, f)) and '-' in f]
    for combined_index in combined_indexes:
        if not any(index["name"] == combined_index for index in pdf_indexes):
            pdf_indexes.append({
                "name": combined_index,
                "directory": os.path.join(db_directory, combined_index)
            })

    # Interactive loop to select indexes and query
    while True:
        print(colored("\nSelect the indexes to query (separate by commas, or type 'all' for all indexes):", 'yellow'))
        for i, index in enumerate(pdf_indexes):
            print(colored(f"{i + 1}. {index['name']}", 'cyan'))

        # Get user input for selected indexes
        selected_indexes = input("Enter the index numbers (or type 0 to exit): ")
        if selected_indexes == '0':
            break
        elif selected_indexes.lower() == 'all':
            indexes_to_query = pdf_indexes
        else:
            selected_indexes = [int(x) - 1 for x in selected_indexes.split(',')]
            indexes_to_query = [pdf_indexes[i] for i in selected_indexes]

        # Load the indexes
        vector_dbs = load_indexes(indexes_to_query, embedding)

        # Ask if the user wants to include resource documents in the response
        include_resources = input("Do you want to include resource documents in the response? (y/n): ").lower() == 'y'

        # Interactive loop to ask for user queries
        while True:
            user_query = input("\nEnter your query (or type 'exit' to change settings or 'quit' to quit): ")
            if user_query.lower() == 'exit':
                break
            elif user_query.lower() == 'quit':
                exit()

            # Run the query and display the response
            response = run_query(user_query, vector_dbs, include_resources)
            print(colored("\nSearch response:", 'green'))
            print(colored(response['result'], 'green'))
            print("\n\n")

            # Only print resources if they were requested
            if include_resources:
                print(colored("Resources:", 'magenta'), response['source_documents'])

if __name__ == "__main__":
    main()
