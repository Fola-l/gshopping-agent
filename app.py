from flask import Flask, request, jsonify
import os
from crewai import Agent, Task, Crew, Process
from serpapi import GoogleSearch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')
PORT = int(os.getenv('PORT', 5000))

# Set OpenAI API key in environment
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def search_with_serpapi(query):
    params = {
        "engine": "google_shopping",
        "q": query,
        "api_key": SERPAPI_KEY,
        "hl": "en",
        "gl": "us"
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return results.get('shopping_results', [])

def process_product_search(product_name, verbose_description):
    # Create refinement agent
    refinement_agent = Agent(
        role='Search Query Refiner',
        goal='Refine a verbose product description to a short, concise search query while capturing the most relevant details (only first 50 words).',
        backstory="You are an expert in refining long product descriptions into short queries that maintain detail while improving search results.",
        verbose=True,
        allow_delegation=False
    )

    # Create refinement task
    refinement_task = Task(
        description=f"Refine the following verbose product description to create a concise search query (limit: 50 words): {verbose_description}",
        expected_output="A short, concise search query (max 50 words).",
        agent=refinement_agent
    )

    # Set up CrewAI
    crew_refinement = Crew(
        agents=[refinement_agent],
        tasks=[refinement_task],
        verbose=True,
        process=Process.sequential
    )

    # Process the search
    refined_query_result = crew_refinement.kickoff()
    refined_description = str(refined_query_result)
    final_search_query = f"{product_name} {refined_description}"
    
    # Perform the search
    return search_with_serpapi(final_search_query)

@app.route('/api/search', methods=['POST'])
def search_product():
    try:
        # Validate request data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        
        data = request.get_json()
        
        # Validate required fields
        if 'title' not in data or 'description' not in data:
            return jsonify({'error': 'Missing required fields: title and description'}), 400
            
        # Extract data
        product_name = data['title']
        verbose_description = data['description']
        
        # Validate field contents
        if not product_name or not verbose_description:
            return jsonify({'error': 'Title and description cannot be empty'}), 400
            
        # Process the search
        results = process_product_search(product_name, verbose_description)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        # Log the error (in a production environment, you'd want proper logging)
        print(f"Error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=PORT) 