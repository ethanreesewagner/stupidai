import os, requests as F
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import time, warnings
from transformers import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set cache directory to a persistent location
os.environ['TRANSFORMERS_CACHE'] = './transformers_cache'

# Pre-download the model and tokenizer
model_name = "albert-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Suppress specific warnings
warnings.filterwarnings('ignore', message=r'.*pooler.dense.weight.*')
warnings.filterwarnings('ignore', message=r'.*pooler.dense.bias.*')
logging.set_verbosity_error()

# Define the Wikimedia search function
def wikimedia_search(query):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action": "query", "list": "search", "srsearch": query, "format": "json"}
    try:
        response = F.get(url, params=params).json()
        return response
    except ValueError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Response content: {response.content}")
        return None

# Define the function to get page extracts
def get_page_extracts(pageids):
    texts = []
    url = "https://en.wikipedia.org/w/api.php"
    for pageid in pageids:
        params = {"action": "query", "prop": "extracts", "explaintext": True, "pageids": pageid, "format": "json"}
        try:
            page_data = F.get(url, params=params).json()
            pages = page_data.get("query", {}).get("pages", {})
            texts.extend(page.get("extract", "") for page in pages.values())
        except ValueError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Response content: {response.content}")
    return texts

# Define the function to split text into chunks
def split_text_into_chunks(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Define the function to answer questions for a list of chunks
def answer_chunk(qa_pipeline, question, chunk):
    try:
        result = qa_pipeline(question=question, context=chunk)
        return result['answer']
    except Exception as e:
        print(f"Error during question answering: {e}")
        return "Error during question answering."

# Define the function to get answers from a question answering model
def get_answers_from_model(question, texts):
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    combined_texts = ' '.join(texts)
    if not combined_texts.strip():
        return "No relevant information found from the scraped links."

    chunks = split_text_into_chunks(combined_texts)
    print(f"Total number of chunks to analyze: {len(chunks)}")

    answers = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(answer_chunk, qa_pipeline, question, chunk) for chunk in chunks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Answering chunks"):
            answers.append(future.result())

    return ' '.join(answers)

# Main function
def main():
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        # Perform Wikimedia searches and collect page IDs
        all_pageids = set()
        results = wikimedia_search(question)
        if results:
            pageids = [result['pageid'] for result in results.get('query', {}).get('search', [])]
            all_pageids.update(pageids)
            time.sleep(2)

        # Get page extracts
        scraped_texts = get_page_extracts(all_pageids)
        print(f"Number of pages extracted: {len(scraped_texts)}")
        if not scraped_texts:
            print("No pages extracted. Skipping question answering.")
            continue

        # Get the answers from the model
        answers = get_answers_from_model(question, scraped_texts)
        print("Answers:", answers)

if __name__ == "__main__":
    main()
