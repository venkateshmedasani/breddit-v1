# breddit-v1

Three Python tools for finding relevant subreddits, scraping posts, and generating AI-powered organic content.

## Setup

### 1. Install Dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_md

### 2. Configure Credentials
Edit the top of each Python file and add your credentials:
REDDIT_CLIENT_ID = "your_client_id_here"
REDDIT_CLIENT_SECRET = "your_client_secret_here"
REDDIT_USER_AGENT = "MyApp/1.0 by u/YourUsername"

### 3. Install Ollama (for content generation only)
Download from https://ollama.ai
ollama pull llama3

## How to Use

### Tool 1: contextsubredditfinder.py
Finds relevant subreddits based on your target audience and topics.

Configure:
WORKFLOW_MODE = "auto_keywords"
BRAND_CONTEXT = {
    "target_customer": "Sales teams, Growth Marketers",
    "intersection_topics": "Lead Generation, Sales Automation"
}

Run: python contextsubredditfinder.py
Output: List of 20+ relevant subreddits

### Tool 2: postscraper.py
Scrapes top posts and comments from a target subreddit.

Configure:
SUBREDDIT_NAME = "SaaS"
POSTS_LIMIT = 100

Run: python postscraper.py
Output: subreddit_data.json with posts and comments

### Tool 3: ContentGen.py
Generates organic Reddit posts based on scraped data.

Prerequisites:
- Run postscraper.py first to create subreddit_data.json
- Start Ollama: ollama serve

Configure:
subreddit_name = "SaaS"
product = "data enrichment tool"
customer_goals = "Find leads faster, automate outreach..."

Run: python ContentGen.py
Output: 2-3 organic post templates (title + body)

## Files
contextsubredditfinder.py - Discovers relevant subreddits using semantic search
postscraper.py - Scrapes posts/comments from target subreddits
ContentGen.py - Generates organic content using LLM analysis
requirements.txt - Python dependencies

## Requirements
- Python 3.8+
- Reddit API account
- Ollama (for content generation)
