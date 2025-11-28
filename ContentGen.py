import json
import nltk
from collections import Counter
import re
import requests


# ---------- Load and Analyze Your Subreddit Data ----------
with open('subreddit_data.json', 'r', encoding='utf-8') as f:
    posts = json.load(f)

titles = [post.get('title', '') for post in posts]
bodies = [post.get('body', '') for post in posts]

# Basic stats for style analysis
all_titles_text = ' '.join(titles)
tokens = nltk.word_tokenize(all_titles_text.lower())
bigrams = Counter(nltk.bigrams(tokens))
most_common_bigrams = bigrams.most_common(10)
avg_title_length = sum(len(t.split()) for t in titles if t) / len(titles)
avg_body_length = sum(len(b.split()) for b in bodies if b) / max(len(bodies), 1)
stop_words = set(nltk.corpus.stopwords.words('english'))
title_words = [w for w in tokens if w.isalpha() and w not in stop_words]
top_words = Counter(title_words).most_common(10)
questions = [t for t in titles if '?' in t]
has_bullets = sum(1 for b in bodies if re.search(r"^\s*[-*]", b, re.MULTILINE))

summary_stats = f"""
- Average title length: {avg_title_length:.1f} words
- Average body length: {avg_body_length:.1f} words
- Most common bigrams in titles: {most_common_bigrams}
- Most common words: {top_words}
- About {len(questions)}/{len(titles)} titles include a question
- {has_bullets} posts have bullet-point lists in body
"""

# ---------- Brand and Customer Context ----------
subreddit_name = "YourSubreddit"
brand = "SampleBrand"
product = "your product here"

# NEW: Customer goals/objectives
customer_goals = "Crustdata is used by customers who are mostly building products rather than running manual workflows—AI SDR tools, recruiting platforms, investor tools, CRM/data teams, and AI agent platforms. AI outbound teams use it to discover new accounts and contacts that match an ICP, detect buying signals in real time (funding, hiring, traffic, posts, reviews), personalize outreach with up-to-date titles and companies, and replace messy stacks of Apollo/Clearbit/LinkedIn scrapers with one API-first data layer. Recruiting platforms use it to power sourcing tools that detect when candidates change jobs, update skills, become poachable, or show intent, and to track hiring trends so they can instantly refresh talent databases. Investors and deal-sourcing platforms use Crustdata to find early-stage companies before they’re widely known, score them using headcount growth, funding, traffic, hiring, reviews, and posts, and monitor portfolios for risk or follow-ons via live signals. Internal sales ops and CRM teams use it to turn “CRM graveyards” into real-time systems of record by enriching and deduping contacts and accounts, pushing watcher signals into CRMs, and removing dependence on multiple enrichment vendors. AI/LLM agent platforms use Crustdata as a structured gateway to the internet so agents can fetch people, company, and signal data without building brittle scrapers or relying on unstructured consumer web search. The main pain points across all these customers are stale data (legacy vendors refresh monthly, missing job changes and new founders entirely), fragmented vendor stacks (one tool for firmographics, another for emails, another for jobs, etc.), lack of real-time event detection (“tell me the second a VP Sales posts about outbound”), missing or shallow data types (most vendors don’t provide posts, reactors, job descriptions, start/end dates, skills, traffic, reviews, technographics), and the burden of maintaining their own scrapers and infrastructure. Many also struggle with poor developer experience from competitors—docs not readable by LLMs, hard-to-use schemas, slow support, and unpredictable credit-based pricing. Competitors fall into a few categories. Static dataset players like Coresignal and People Data Labs offer massive but mostly monthly-refreshed databases with no real-time search or webhook-driven events. Enrichment/contact tools like Clearbit, Apollo, and Lusha focus on manual sales teams, not product teams or agents, and mainly offer firmographics + contact info rather than posts, reactors, reviews, traffic, news, and multi-source intelligence. LinkedIn-focused vendors like Proxycurl and MixRank enrich profiles but don’t cover Crustdata’s full scope—multi-source company profiles from 15–16+ sources, social posts + reactors, reviews, news, traffic, and event-driven Watcher APIs. Many prospects also compare Crustdata to “DIY scraping,” but switch because Crustdata eliminates crawling, parsing, maintenance, and compliance headaches. Crustdata’s core differentiators are real-time enrichment instead of monthly refresh, real-time search APIs instead of static datasets, multi-source breadth (posts, reactors, traffic, reviews, technographics, jobs, news), structured APIs built for AI agents, and webhook-driven Watcher events that let customers detect changes as they happen. Overall, Crustdata positions itself not as a traditional data vendor but as a real-time B2B data layer for AI-native products—combining real-time search, live enrichment, event-based signals, and multi-source intelligence so AI SDRs, recruiters, investors, and internal tools can act on what’s happening right now, not last month."

# ---------- Enhanced Prompt with Customer Goals ----------
ollama_prompt = f"""
You are a Reddit content strategist.

Here are style and content statistics from top posts in r/{subreddit_name}:
{summary_stats}

CUSTOMER CONTEXT:
The target audience is trying to achieve the following:
{customer_goals}

TASK:
Create 2–3 organic post templates for a user who is sharing experience or seeking advice about a product like '{product}', possibly inspired by '{brand}'. The user is *not* affiliated with the brand and should not sound as if they're advertising.

IMPORTANT GUIDELINES:
- Posts must align with the customer's goals and pain points mentioned above (e.g., struggling with lead generation, looking for automation, seeking efficiency)
- Use community-relevant style and structure (e.g., casual questions, storytelling, bullet points)
- Focus on the theme of the subreddit and the general topics being spoken about in the subreddit.
- Do NOT use the brand or product name directly in the title or body
- Avoid promotional or marketing language. Don't say "buy," "try," or mention specific brands or URLs
- Instead, focus on:
  * Honest personal experience related to customer goals
  * Indirect references ("I tried something new," "found a way to speed up X")
  * Natural discussion-starters about challenges the customer faces
  * Questions about how others solve these problems
- Posts should sound like a genuine Redditor dealing with real problems (e.g., "struggling to find quality leads," "tired of manual outreach," "looking for efficiency")
- Each post should invite real engagement, like responses, opinions, or stories from other users
- Frame posts around the customer's struggles, not the product's features
- Make the post body longer around 100-200 words with necessary content to describe pain points and problems the customer is facing in more detail.
- Make sure the body of the post matches the tone, language, and formatting of these scraped posts, tells a first-person story about a specific unresolved problem that people here commonly face, does not mention or hint at any product or brand, and ends by asking the community what tools, workflows, or solutions they recommend (so that a suitable product could later be suggested in the comments as one of the possible answers)

OUTPUT FORMAT:
Output only the posts — with "Title:" and "Body:" labels, no explanations, no direct plugs.

EXAMPLE OF ORGANIC STYLE:
Title: How do you all manage to scale outreach without burning out?
Body: I've been manually reaching out to prospects for months and it's exhausting. Curious how others in similar roles handle this at scale without losing the personal touch.

Now, generate 2-3 templates that address the customer's goals naturally. Remember: keep it natural, organic, and related to their real challenges.
"""

# ---------- Send Prompt to Ollama (Llama3) Server ----------
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3", "prompt": ollama_prompt, "stream": False}
)

print("\nGenerated Templates:\n")
print(response.json()['response'])
