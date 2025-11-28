import praw
import json
import time
import random
from datetime import datetime

# ---- CONFIGURATION ----
REDDIT_CLIENT_ID = "placeholder"
REDDIT_CLIENT_SECRET = "placeholder"
REDDIT_USER_AGENT = "placeholder"
SUBREDDIT_NAME ="Recruitment"  # Change this to your target
POSTS_LIMIT = 100  # Reddit/PRAW hard API limit per filter

# ---- INITIALIZE REDDIT INSTANCE ----
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

def scrape_subreddit_top_posts(subreddit_name, posts_limit=1000):
    posts_all = []
    subreddit = reddit.subreddit(subreddit_name)
    print(f"Accessing subreddit '{subreddit_name}' for top posts...")
    print(f"Parameters: time_filter='year', limit={posts_limit}")
    for post in subreddit.top(time_filter="year", limit=posts_limit):
        post.comments.replace_more(limit=None)
        post_data = {
            "post_id": post.id,
            "title": post.title,
            "selftext": post.selftext,
            "url": post.url,
            "score": post.score,
            "upvote_ratio": post.upvote_ratio,
            "num_comments": post.num_comments,
            "created_utc": datetime.fromtimestamp(post.created_utc).isoformat(),
            "author": str(post.author),
            "flair": post.link_flair_text,
            "permalink": f"https://reddit.com{post.permalink}",
            "comments_data": []
        }
        for comment in post.comments.list():
            post_data["comments_data"].append({
                "comment_id": comment.id,
                "parent_id": comment.parent_id,
                "body": comment.body,
                "author": str(comment.author),
                "score": comment.score,
                "created_utc": datetime.fromtimestamp(comment.created_utc).isoformat(),
                "is_submitter": comment.is_submitter
            })
        posts_all.append(post_data)
        # ---- RANDOM DELAY BETWEEN POSTS ----
        sleep_time = random.uniform(0.5, 1.5)
        time.sleep(sleep_time)

        for idx, post in enumerate(subreddit.top(time_filter="year", limit=posts_limit), 1):
            print(f"[{idx}/{posts_limit}] Scraped post: {post.title[:40]}")


    return posts_all

def save_to_json(data, subreddit_name):
    filename = f"subreddit_data.json"
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} posts to {filename}")

if __name__ == "__main__":
    posts = scrape_subreddit_top_posts(SUBREDDIT_NAME, POSTS_LIMIT)
    save_to_json(posts, SUBREDDIT_NAME)
