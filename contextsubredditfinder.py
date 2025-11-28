import praw
import requests
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy


# --- CONFIG ---
REDDIT_CLIENT_ID = "placeholder"
REDDIT_CLIENT_SECRET = "placeholder"
REDDIT_USER_AGENT = "placeholder"

# Choose workflow mode: "manual", "brand_context", or "auto_keywords"
WORKFLOW_MODE = "auto_keywords"  # CHANGE THIS to switch workflows

# WORKFLOW 1 & 2: Manual keywords list
KEYWORDS = ["B2B Data", "Sales Automation", "Lead Generation", "GTM", "SaaS"]

# WORKFLOW 3: Brand context (columns B and D)
BRAND_CONTEXT = {
    "target_customer": "Sales teams, GTM teams, Growth Marketers, SaaS Founders",
    "intersection_topics": "B2B Data, Sales Automation, GTM, Growth Hacking, Lead Generation, Cold Emailing, RevOps"
}

DESIRED_COUNT = 20
RELATED_COUNT = 20

MIN_KEYWORD_POSTS = 2
MIN_RATIO = 0.02

# Stricter semantic threshold for auto_keywords mode
SEMANTIC_SIM_THRESHOLD = 0.35  # Default
STRICT_SEMANTIC_THRESHOLD = 0.45  # For auto_keywords workflow


# ============================================
# BRAND CONTEXT CLASS
# ============================================

class BrandContext:
    """Stores brand information from columns B and D"""
    def __init__(self, target_customer, intersection_topics):
        self.target_customer = target_customer
        self.intersection_topics = intersection_topics


# ============================================
# KEYWORD GENERATION FROM BRAND CONTEXT
# ============================================

def generate_keywords_from_brand(brand_context):
    """
    WORKFLOW 3: Extract keywords from target customer and intersection topics.
    Generates search terms without using spaCy variants.
    """
    keywords = []
    
    # Extract intersection topics (Column D)
    topics = [t.strip() for t in brand_context.intersection_topics.split(',')]
    keywords.extend(topics)
    
    # Extract target customer terms (Column B)
    customers = [c.strip() for c in brand_context.target_customer.split(',')]
    keywords.extend(customers)
    
    # Generate variations
    additional_keywords = []
    
    # Add singular/plural variations
    for kw in keywords:
        if kw.endswith('s') and len(kw) > 4:
            additional_keywords.append(kw[:-1])
        else:
            additional_keywords.append(kw + 's')
    
    # Add combined terms (customer + topic)
    for customer in customers[:2]:  # First 2 customer types
        for topic in topics[:2]:  # First 2 topics
            combined = f"{customer} {topic}"
            additional_keywords.append(combined)
    
    # Add "how to" variations for topics
    for topic in topics[:3]:
        additional_keywords.append(f"how to {topic.lower()}")
        additional_keywords.append(f"{topic.lower()} tools")
    
    # Combine and deduplicate
    all_keywords = list(set(keywords + additional_keywords))
    
    print(f"Generated keywords from brand context:")
    print(f"  - Core topics (Column D): {topics}")
    print(f"  - Customer terms (Column B): {customers}")
    print(f"  - Total keywords: {len(all_keywords)}")
    print(f"  - Sample: {all_keywords[:10]}")
    
    return all_keywords


# --- spaCy keyword variant expansion ---
def get_word_variants(keywords):
    """Generate variants for multiple keywords using spaCy"""
    nlp = spacy.load('en_core_web_md')
    all_variants = set(keywords)  # Start with original keywords
    
    for keyword in keywords:
        token = nlp(keyword)
        similar_words = [w.text for w in nlp.vocab if w.has_vector and w.is_lower and w.prob >= -15]
        
        for w in similar_words:
            if token.similarity(nlp(w)) > 0.50 and w not in keyword:
                all_variants.add(w)
            if len(all_variants) >= 25:  # Limit for runtime
                break
    
    variants_list = list(all_variants)
    print(f"spaCy-generated keyword variants ({len(variants_list)} total): {variants_list}")
    return variants_list


# --- INIT REDDIT & BERT ---
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

model = SentenceTransformer('all-MiniLM-L6-v2')


def fetch_posts(keywords, max_posts=150):
    """Fetch posts for multiple keywords"""
    all_posts = []
    seen_ids = set()
    
    for keyword in keywords[:10]:  # Limit to first 10 keywords
        print(f"Searching Reddit posts for '{keyword}'...")
        try:
            for post in reddit.subreddit("all").search(keyword, sort="relevance", limit=max_posts):
                if post.id not in seen_ids:
                    all_posts.append(post)
                    seen_ids.add(post.id)
        except Exception as e:
            print(f"Error searching '{keyword}': {e}")
    
    print(f"Found {len(all_posts)} unique posts across all keywords.")
    return all_posts


def ordered_unique_subs(posts):
    ordered_subs = []
    seen = set()
    for p in posts:
        subname = p.subreddit.display_name
        if subname not in seen:
            ordered_subs.append(subname)
            seen.add(subname)
    print(f"Candidate subreddits (unique, ordered): {len(ordered_subs)} subreddits")
    return ordered_subs


def check_subreddit_relevance(
    subreddit,
    keyword_list,
    keyword_embeddings,
    min_posts=MIN_KEYWORD_POSTS,
    min_ratio=MIN_RATIO,
    use_semantics=True,
    sim_threshold=SEMANTIC_SIM_THRESHOLD
):
    """Check subreddit relevance with configurable semantic threshold"""
    sub_inst = reddit.subreddit(subreddit)
    matching_count = 0
    total_count = 0
    
    for p in sub_inst.new(limit=100):
        total_count += 1
        post_text = (p.title or "") + " " + (p.selftext or "")
        match_found = False

        # Traditional keyword match
        for kw in keyword_list:
            if kw.lower() in post_text.lower():
                match_found = True
                break

        # Semantic context match
        if not match_found and use_semantics:
            post_emb = model.encode(post_text)
            sims = cosine_similarity(np.array([post_emb]), keyword_embeddings)
            if np.max(sims) > sim_threshold:
                match_found = True

        if match_found:
            matching_count += 1

    print(f"r/{subreddit}: {matching_count} relevant posts / {total_count} recent posts")
    if total_count == 0:
        return False
    
    ratio = matching_count / total_count
    passes = matching_count >= min_posts and ratio >= min_ratio
    print(f"r/{subreddit}: relevance passes: {passes} (min_posts={min_posts}, min_ratio={min_ratio})")
    return passes


def get_communities_section_subs(keyword, exclude_subs, max_subs=25):
    # Use Reddit's public API for communities tab results
    url = f"https://www.reddit.com/subreddits/search.json?q={keyword}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    subs = []
    exclude_set = set([s.lower() for s in exclude_subs])
    try:
        data = resp.json()
        for child in data.get('data', {}).get('children', []):
            subname = child.get('data', {}).get('display_name', '')
            if subname and subname.lower() not in exclude_set and subname not in subs:
                subs.append(subname)
            if len(subs) == max_subs:
                break
    except Exception as e:
        print(f"Error retrieving communities API: {e}")
    return subs


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print(f"SUBREDDIT FINDER - Workflow: {WORKFLOW_MODE.upper()}")
    print("=" * 70)
    
    # Initialize variables
    keyword_variants = []
    semantic_threshold = SEMANTIC_SIM_THRESHOLD
    
    # ============================================
    # WORKFLOW SELECTION
    # ============================================
    
    if WORKFLOW_MODE == "manual":
        # WORKFLOW 1: Manual keywords with spaCy variants
        print("\n📌 Workflow 1: Manual Keywords with spaCy Expansion")
        print(f"Input keywords: {KEYWORDS}")
        keyword_variants = get_word_variants(KEYWORDS)
        
    elif WORKFLOW_MODE == "brand_context":
        # WORKFLOW 2: Brand context with spaCy variants
        print("\n📌 Workflow 2: Brand Context with spaCy Expansion")
        brand = BrandContext(
            target_customer=BRAND_CONTEXT["target_customer"],
            intersection_topics=BRAND_CONTEXT["intersection_topics"]
        )
        print(f"Target Customer: {brand.target_customer}")
        print(f"Intersection Topics: {brand.intersection_topics}")
        
        # Extract base keywords from brand context
        base_keywords = []
        base_keywords.extend([t.strip() for t in brand.intersection_topics.split(',')])
        base_keywords.extend([c.strip() for c in brand.target_customer.split(',')])
        
        keyword_variants = get_word_variants(base_keywords)
        
    elif WORKFLOW_MODE == "auto_keywords":
        # WORKFLOW 3: Auto-generate keywords from brand context (NO spaCy)
        print("\n📌 Workflow 3: Auto-Generated Keywords from Brand Context")
        print("🔒 STRICT SEMANTIC MATCHING ENABLED")
        
        brand = BrandContext(
            target_customer=BRAND_CONTEXT["target_customer"],
            intersection_topics=BRAND_CONTEXT["intersection_topics"]
        )
        print(f"Target Customer (Column B): {brand.target_customer}")
        print(f"Intersection Topics (Column D): {brand.intersection_topics}")
        print()
        
        # Generate keywords WITHOUT spaCy variants
        keyword_variants = generate_keywords_from_brand(brand)
        
        # Use stricter semantic threshold
        semantic_threshold = STRICT_SEMANTIC_THRESHOLD
        print(f"\n⚙️  Using STRICT semantic threshold: {semantic_threshold} (default: {SEMANTIC_SIM_THRESHOLD})")
    
    else:
        print(f"❌ Unknown workflow mode: {WORKFLOW_MODE}")
        print("   Valid modes: 'manual', 'brand_context', 'auto_keywords'")
        exit(1)
    
    # ============================================
    # COMMON WORKFLOW (All modes)
    # ============================================
    
    print("\n" + "=" * 70)
    print("EXECUTING SUBREDDIT DISCOVERY")
    print("=" * 70)
    
    # Create embeddings
    keyword_embeddings = model.encode(keyword_variants)
    print(f"\n✓ Created embeddings for {len(keyword_variants)} keywords")
    
    # Fetch posts
    posts = fetch_posts(keyword_variants, max_posts=150)
    candidate_subs = ordered_unique_subs(posts)

    # Check relevance
    print(f"\nProcessing candidate subreddits for relevance threshold...")
    print(f"Semantic threshold: {semantic_threshold}")
    final_subs = []
    checked = set()
    
    for sub in candidate_subs:
        if len(final_subs) >= DESIRED_COUNT:
            break
        if sub in checked:
            continue
        checked.add(sub)
        
        if check_subreddit_relevance(
            sub, 
            keyword_variants, 
            keyword_embeddings,
            sim_threshold=semantic_threshold  # Use workflow-specific threshold
        ):
            final_subs.append(sub)
            print(f"Added r/{sub} to final list.\n")

    # Display results
    print(f"\n{'=' * 70}")
    print(f"🏆 Top {len(final_subs)} relevant subreddits:")
    print(f"{'=' * 70}")
    for idx, sub in enumerate(final_subs, 1):
        print(f"{idx}. r/{sub}")

    # Find additional subreddits from Communities tab
    primary_keyword = keyword_variants[0] if keyword_variants else "reddit"
    communities_subs = get_communities_section_subs(primary_keyword, final_subs, max_subs=RELATED_COUNT)
    
    print(f"\nUp to {RELATED_COUNT} additional subreddits from Communities section:")
    for idx, sub in enumerate(communities_subs, 1):
        print(f"{idx}. r/{sub}")

    print("\n" + "=" * 70)
    print("✅ Done.")
    print("=" * 70)
