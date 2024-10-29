import os
from crewai import Agent, Task, Crew, Process
from serpapi import GoogleSearch  # SerpAPI's Google Shopping tool
from dotenv import load_dotenv

load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SERPAPI_KEY = os.getenv('SERPAPI_KEY')

# Set OpenAI API key in environment
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Step 1: Get user input for product name and verbose description
product_name = input("Enter the product name (e.g., Pot Scrubber): ")
verbose_description = input(f"Enter the product description and features for {product_name}: ")

# Step 2: Define the refining agent to make the description short and concise using CrewAI
refinement_agent = Agent(
    role='Search Query Refiner',
    goal='Refine a verbose product description to a short, concise search query while capturing the most relevant details (only first 50 words).',
    backstory="""You are an expert in refining long product descriptions into short queries that maintain detail while improving search results.""",
    verbose=True,
    allow_delegation=False
)

# Task to refine the verbose product description (focusing on the first 50 words) for a better search query
refinement_task = Task(
    description=f"Refine the following verbose product description to create a concise search query (limit: 50 words): {verbose_description}",
    expected_output="A short, concise search query (max 50 words).",
    agent=refinement_agent
)

# Step 3: Set up CrewAI to process the refinement task
crew_refinement = Crew(
    agents=[refinement_agent],
    tasks=[refinement_task],
    verbose=True,
    process=Process.sequential
)

# Kick off the refinement task to get the refined query
refined_query_result = crew_refinement.kickoff()
refined_description = str(refined_query_result)

# Construct the final query with product name first
final_search_query = f"{product_name} {refined_description}"

# Step 4: Search with SerpAPI Google Shopping API
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

# Perform the search using SerpAPI with the final query
shopping_results = search_with_serpapi(final_search_query)

# Step 5: Sort search results by price from most expensive to cheapest
def extract_price(result):
    try:
        return float(result.get('extracted_price', 0))
    except ValueError:
        return 0  # Handle cases where the price is not valid

# Sorting the shopping results based on price
sorted_results = sorted(shopping_results, key=extract_price, reverse=True)

# Step 6: Pass sorted results to a refining agent using CrewAI
response_refining_agent = Agent(
    role='Response Refiner',
    goal='Extract relevant information such as titles, prices, and links from the sorted JSON response.',
    backstory="""You are skilled at extracting the most important details from the shopping results, such as titles, prices, and links.""",
    verbose=True,
    allow_delegation=False
)

# Task to refine the sorted shopping results
response_refining_task = Task(
    description=f"Extract relevant information (titles, prices, and links) from the following sorted search results: {sorted_results}",
    expected_output="A refined list of titles, prices, and links.",
    agent=response_refining_agent
)

# Set up CrewAI to refine the sorted results
crew_response_refining = Crew(
    agents=[response_refining_agent],
    tasks=[response_refining_task],
    verbose=True,
    process=Process.sequential
)

# Kick off the response refining task
refined_results = crew_response_refining.kickoff()

# Step 7: Pass the refined results to a web-checking agent using CrewAI
web_checking_agent = Agent(
    role='Web Checking Agent',
    goal='Verify that the links from the search results are valid and match the product description.',
    backstory="""You are responsible for verifying that the links lead to relevant products and are accessible.""",
    verbose=True,
    allow_delegation=False
)

web_checking_task = Task(
    description=f"Verify the validity of the links from the following refined search results: {refined_results}",
    expected_output="A list of valid and accessible links.",
    agent=web_checking_agent
)

# Set up CrewAI to validate the links
crew_web_checking = Crew(
    agents=[web_checking_agent],
    tasks=[web_checking_task],
    verbose=True,
    process=Process.sequential
)

# Kick off the web-checking task
validated_links = crew_web_checking.kickoff()

# Step 8: Pass the final results to a writing agent to generate the final report
write_agent = Agent(
    role='Write Agent',
    goal='Generate a final report comparing the original query with the refined search results, highlighting matching and non-matching results.',
    backstory="""You are skilled at comparing search results with the original query and generating detailed reports.""",
    verbose=True,
    allow_delegation=False
)

write_task = Task(
    description=f"Compare the original product query with the refined search results. Highlight what matches and what does not.",
    expected_output="A detailed report on the search results.",
    agent=write_agent
)

# Set up CrewAI for the final write task
crew_write = Crew(
    agents=[write_agent],
    tasks=[write_task],
    verbose=True,
    process=Process.sequential
)

# Kick off the final writing task to generate the report
final_report = crew_write.kickoff()

# Output the final report to the user
print("Final Report:\n")
print(final_report)