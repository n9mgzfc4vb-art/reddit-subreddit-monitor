# ========================================================================
# Reddit Subreddit Monitor
# ========================================================================
# Personal project to explore AI/ML tooling on public Reddit data.
# Fetches posts and comments from a target subreddit, applies topic
# tags post hoc, and exports to CSV for downstream analysis.
#
# Architecture: Full capture of all comments (no keyword filtering).
# Tag families applied post hoc for filtering and analysis.
#
# Output: One CSV, one row per comment, all metadata + boolean tag flags.
# ========================================================================

# -------------------------------
# STEP 0: Install dependencies
# -------------------------------
# !pip install praw pandas --quiet  # Uncomment if running in a notebook

# -------------------------------
# STEP 1: Imports
# -------------------------------
import praw
import logging
import pandas as pd
import re
import time
import warnings
from datetime import datetime, timezone

warnings.filterwarnings('ignore', category=DeprecationWarning)

# -------------------------------
# STEP 2: Suppress PRAW warnings
# -------------------------------
logging.getLogger("praw").setLevel(logging.ERROR)
logging.getLogger("prawcore").setLevel(logging.ERROR)

# -------------------------------
# STEP 3: Reddit API Setup
# -------------------------------
# Store your credentials in environment variables rather than
# hardcoding them here. See README for setup instructions.
import os

reddit = praw.Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
    user_agent="subreddit_monitor_personal"
)

# -------------------------------
# STEP 4: Configuration
# -------------------------------
SUBREDDIT = "awardtravel"       # Target subreddit (no r/ prefix)
POSTS_LIMIT = 1000              # Max posts to fetch
REQUEST_DELAY = 2.0             # Seconds between requests (respect rate limits)
VERSION_LABEL = "MOST_RECENT"  # Label for output file versioning

# -------------------------------
# STEP 5: Tag Families
#
# Every non-junk comment is captured regardless of keywords.
# These tag families flag comments by topic for downstream filtering.
# Matching is case-insensitive via word-boundary regex.
#
# Airline names are spelled out rather than using IATA codes to avoid
# false positives in casual text.
# -------------------------------

TAG_FAMILIES = {

    # -----------------------------------------------------------------
    # 1. LOYALTY PROGRAM
    #    Comments about airline loyalty programs — tiers, earning,
    #    redemption, credit cards, and program changes.
    # -----------------------------------------------------------------
    "loyalty_program": [
        # Program names and tiers
        "Mileage Plan", "MVP", "MVP Gold", "MVP Gold 75K",
        "MVP Gold 100K", "elite status", "status", "tier",
        "status match", "status challenge",
        # Earn and burn mechanics
        "miles", "award", "award flight", "redemption", "redeem",
        "saver award", "saver fare", "earning rate", "earn rate",
        "bonus miles", "elite miles", "EQM",
        "companion fare", "companion pass", "companion certificate",
        "upgrade", "upgrades", "first class upgrade",
        "wallet", "award wallet",
        # Credit card / co-brand
        "Alaska card", "Alaska credit card", "Bank of America",
        "annual fee", "sign-up bonus", "signup bonus",
        "lounge access", "lounge",
        # Program changes / reactions
        "devaluation", "devalue", "program change", "new program",
        "used to be better", "worse than before", "getting worse",
        "still worth", "best program", "love the program",
        "loyal", "loyalty",
    ],

    # -----------------------------------------------------------------
    # 2. FLIGHT EXPERIENCE
    #    Comments about the flying product — cabin, service, reliability,
    #    digital tools, and route network.
    # -----------------------------------------------------------------
    "flight_experience": [
        # Hard product
        "seat", "legroom", "premium class", "first class", "premium",
        "economy", "main cabin", "saver", "cabin", "recline",
        "power outlet", "wifi", "entertainment", "IFE",
        # Soft product
        "service", "crew", "flight attendant", "gate agent",
        "friendly", "helpful", "rude", "customer service",
        "boarding", "on time", "delay", "delayed", "cancelled",
        "cancellation", "rebooking", "rebooked",
        # Food and beverage
        "snack", "food", "meal", "drink", "coffee",
        # App and digital
        "app", "website", "booking", "check in", "mobile",
        # Route network
        "route", "nonstop", "hub", "Seattle", "West Coast",
        "partner", "oneworld",
    ],

    # -----------------------------------------------------------------
    # 3. COMPETITOR MENTIONS
    #    Comments that mention competing airlines and their programs.
    #    Useful for understanding how Alaska is benchmarked against peers.
    # -----------------------------------------------------------------
    "competitor_mentions": [
        # Major US carriers + programs
        "Delta", "SkyMiles", "Medallion",
        "United", "MileagePlus", "Premier",
        "American", "AAdvantage", "American Airlines",
        "Southwest", "Rapid Rewards",
        "JetBlue", "TrueBlue",
        "Spirit", "Frontier",
        # Alliances
        "oneworld", "Star Alliance", "SkyTeam",
        # International carriers
        "Cathay", "British Airways", "Avios",
        "Emirates", "Singapore Airlines", "Qantas",
        # Switching language
        "switch", "switched", "switching",
        "moved to", "better than", "instead of",
        "used to fly", "trying", "defected",
        "went to", "going to", "left Alaska",
        "done with Alaska", "over Alaska",
        "compared to", "comparison", "versus",
    ],

    # -----------------------------------------------------------------
    # 4. VALUE & PRICE
    #    Comments about price sensitivity and perceived value.
    # -----------------------------------------------------------------
    "value_and_price": [
        # Positive value signals
        "worth it", "great value", "great deal", "best value",
        "love Alaska", "love this airline", "fan", "best airline",
        "underrated", "hidden gem", "gem",
        "keeps me coming back", "why I fly Alaska",
        "punches above", "bang for the buck",
        # Negative value signals
        "overpriced", "expensive", "price", "pricey",
        "not worth", "rip off", "ripoff", "fare",
        "baggage fee", "bag fee", "fees",
        "getting greedy", "cash grab",
        # Comparative value
        "cheaper on", "cheaper with", "better deal",
        "same price", "for that price",
        "basic economy", "no frills",
    ],

    # -----------------------------------------------------------------
    # 5. BRAND SENTIMENT
    #    Comments reflecting emotional attachment, regional identity,
    #    and brand affinity.
    # -----------------------------------------------------------------
    "brand_sentiment": [
        # Brand attachment
        "obsessed", "die hard", "diehard", "ride or die",
        "always fly Alaska", "Alaska fan", "love Alaska",
        "proud", "represent", "my airline",
        # Regional identity
        "PNW", "Pacific Northwest", "Seattle", "Portland",
        "West Coast", "Alaska roots", "hometown",
        # Heritage
        "Virgin America", "miss Virgin", "used to be Virgin",
        "Alaska Air Group", "merger", "acquisition",
        "Horizon", "Horizon Air",
        # Community and culture
        "community", "culture", "vibe", "brand",
        "Eskimo", "livery", "tail",
        # Trust
        "trust", "reliable", "consistent", "dependable",
        "never let me down", "always on time",
        "safety", "safe",
    ],

    # -----------------------------------------------------------------
    # 6. PROGRAM FRICTION
    #    Comments about pain points in the loyalty experience —
    #    earning, redemption, partner issues, and customer service.
    # -----------------------------------------------------------------
    "program_friction": [
        # Earning frustrations
        "hard to earn", "earn enough", "not earning",
        "qualification", "qualify", "requalify", "rollover",
        "elite qualifying", "segments",
        # Redemption frustrations
        "availability", "award availability", "no availability",
        "blackout", "can't redeem", "impossible to book",
        "award chart", "dynamic pricing", "revenue based",
        # Partner issues
        "partner award", "partner airline", "partner booking",
        "transfer", "transfer partner",
        "Cathay Pacific", "British Airways", "Finnair",
        "Emirates", "Qantas", "oneworld partner",
        # Customer service friction
        "phone", "hold", "wait time", "agent",
        "resolution", "complaint", "frustrated", "frustrating",
        "disappointed", "disappointing", "unacceptable",
        # Policy friction
        "change fee", "cancel", "refund", "voucher", "credit",
        "expiration", "expire", "expired",
    ],
}


# -------------------------------
# STEP 6: Utility Functions
# -------------------------------
def is_bot_or_junk(text):
    """Filter out bot comments and very short comments."""
    if pd.isna(text) or len(str(text).strip()) < 20:
        return True
    text_lower = str(text).lower()
    junk_indicators = [
        'i am a bot',
        'this action was performed automatically',
        'contact the moderators',
        'minimum account age and karma',
        'your submission has been removed',
        'hello! this is a comment to let you know',
    ]
    return any(indicator in text_lower for indicator in junk_indicators)


def tag_text(text):
    """
    Run all tag families against a piece of text.
    Returns a dict of {family_name: [matched_signals]} for each family,
    plus a boolean for whether ANY family matched.
    Matching is case-insensitive.
    """
    if pd.isna(text):
        return False, {}

    text_lower = str(text).lower()
    results = {}

    for family_name, signals in TAG_FAMILIES.items():
        matched = []
        for signal in signals:
            pattern = r'\b' + re.escape(signal.lower()) + r'\b'
            if re.search(pattern, text_lower):
                matched.append(signal)
        if matched:
            results[family_name] = matched

    any_hit = len(results) > 0
    return any_hit, results


def utc_to_iso(utc_timestamp):
    """Convert Reddit UTC timestamp to ISO format string."""
    return datetime.fromtimestamp(utc_timestamp, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# ---------------------------------------------------------------
# STEP 7: Scrape
# ---------------------------------------------------------------
all_comments = []
posts_processed = 0
comments_scanned = 0
comments_captured = 0
tag_counts = {family: 0 for family in TAG_FAMILIES}
start_time = time.time()

print("=" * 70)
print(f"Reddit Subreddit Monitor — r/{SUBREDDIT} — {VERSION_LABEL}")
print("=" * 70)
print(f"Target: r/{SUBREDDIT}")
print(f"Sort: new")
print(f"Posts limit: {POSTS_LIMIT}")
print(f"Capture mode: ALL comments (no keyword filtering)")
print(f"Tagging: {len(TAG_FAMILIES)} families applied post hoc (case-insensitive)")
print(f"Request delay: {REQUEST_DELAY}s")
print("=" * 70)

try:
    subreddit = reddit.subreddit(SUBREDDIT)

    for submission in subreddit.new(limit=POSTS_LIMIT):
        try:
            post_title = submission.title
            post_body = submission.selftext if submission.selftext else ""
            post_timestamp = utc_to_iso(submission.created_utc)
            post_score = submission.score
            post_num_comments = submission.num_comments
            post_url = submission.url
            post_flair = submission.link_flair_text if submission.link_flair_text else ""
            posts_processed += 1

            # Tag the post itself (title + body)
            post_text_combined = f"{post_title} {post_body}"
            post_tagged, post_tag_matches = tag_text(post_text_combined)

            # Progress every 100 posts
            if posts_processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {posts_processed}/{POSTS_LIMIT} posts | "
                      f"{comments_captured:,} comments captured | "
                      f"Elapsed: {elapsed / 60:.1f}m")

            # Expand full comment tree
            submission.comments.replace_more(limit=0)

            for comment in submission.comments.list():
                comments_scanned += 1

                if is_bot_or_junk(comment.body):
                    continue

                comments_captured += 1

                # Tag the comment
                comment_tagged, comment_tag_matches = tag_text(comment.body)

                # Increment per-family counters
                for family in comment_tag_matches:
                    tag_counts[family] += 1

                all_comments.append({
                    # -- Version --
                    "version": VERSION_LABEL,
                    # -- Post metadata --
                    "subreddit": SUBREDDIT,
                    "post_title": post_title,
                    "post_body": post_body,
                    "post_timestamp": post_timestamp,
                    "post_score": post_score,
                    "post_num_comments": post_num_comments,
                    "post_url": post_url,
                    "post_flair": post_flair,
                    "post_tagged": post_tagged,
                    "post_tag_matches": post_tag_matches if post_tagged else {},
                    # -- Comment data --
                    "comment": comment.body,
                    "comment_timestamp": utc_to_iso(comment.created_utc),
                    "comment_score": comment.score,
                    "comment_parent_id": comment.parent_id,
                    # -- Comment tags --
                    "comment_tagged": comment_tagged,
                    "comment_tag_matches": comment_tag_matches if comment_tagged else {},
                    # Per-family boolean flags for easy filtering
                    "tag_loyalty_program": "loyalty_program" in comment_tag_matches,
                    "tag_flight_experience": "flight_experience" in comment_tag_matches,
                    "tag_competitor_mentions": "competitor_mentions" in comment_tag_matches,
                    "tag_value_and_price": "value_and_price" in comment_tag_matches,
                    "tag_brand_sentiment": "brand_sentiment" in comment_tag_matches,
                    "tag_program_friction": "program_friction" in comment_tag_matches,
                })

            time.sleep(REQUEST_DELAY)

        except Exception as e:
            continue

except Exception as e:
    print(f"\nERROR accessing r/{SUBREDDIT}: {e}")

# ---------------------------------------------------------------
# STEP 8: Summary
# ---------------------------------------------------------------
elapsed = time.time() - start_time
print("\n" + "=" * 70)
print(f"{VERSION_LABEL} — COMPLETE")
print("=" * 70)
print(f"Posts processed:    {posts_processed:,}")
print(f"Comments scanned:   {comments_scanned:,}")
print(f"Comments captured:  {comments_captured:,}")
print(f"  ├─ Loyalty program:      {tag_counts.get('loyalty_program', 0):,}")
print(f"  ├─ Flight experience:    {tag_counts.get('flight_experience', 0):,}")
print(f"  ├─ Competitor mentions:  {tag_counts.get('competitor_mentions', 0):,}")
print(f"  ├─ Value and price:      {tag_counts.get('value_and_price', 0):,}")
print(f"  ├─ Brand sentiment:      {tag_counts.get('brand_sentiment', 0):,}")
print(f"  └─ Program friction:     {tag_counts.get('program_friction', 0):,}")
print(f"Runtime: {elapsed / 60:.1f} minutes")

if len(all_comments) > 0:
    timestamps = [c["post_timestamp"] for c in all_comments]
    print(f"Date range: {min(timestamps)} → {max(timestamps)}")
print("=" * 70)

# ---------------------------------------------------------------
# STEP 9: Export
# ---------------------------------------------------------------
df = pd.DataFrame(all_comments)

if len(df) == 0:
    print("\nWARNING: No comments captured. Check subreddit accessibility.")
else:
    output_path = f"reddit_{SUBREDDIT}_{VERSION_LABEL.lower()}.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✓ Exported → {output_path}")
    print(f"  ({len(df):,} comments)")
    print(f"\nColumns:")
    print(f"  version, subreddit, post_title, post_body, post_timestamp,")
    print(f"  post_score, post_num_comments, post_url, post_flair,")
    print(f"  post_tagged, post_tag_matches,")
    print(f"  comment, comment_timestamp, comment_score, comment_parent_id,")
    print(f"  comment_tagged, comment_tag_matches,")
    print(f"  tag_loyalty_program, tag_flight_experience,")
    print(f"  tag_competitor_mentions, tag_value_and_price,")
    print(f"  tag_brand_sentiment, tag_program_friction")
    print("=" * 70)
