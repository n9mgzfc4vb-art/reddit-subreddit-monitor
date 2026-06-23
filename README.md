# Reddit Subreddit Monitor

A personal project to explore AI/ML tooling on public Reddit data.

## What it does

Fetches posts and comments from a target subreddit using the Reddit API,
applies topic tags post hoc via keyword matching, and exports results to
CSV for analysis and experimentation.

## Tech stack

- Python
- [PRAW](https://praw.readthedocs.io/) (Python Reddit API Wrapper)
- pandas

## Setup

1. Clone this repo
2. Install dependencies: `pip install praw pandas`
3. Set your Reddit API credentials as environment variables:

```bash
export REDDIT_CLIENT_ID=your_client_id
export REDDIT_CLIENT_SECRET=your_client_secret
```

4. Edit `reddit_subreddit_monitor.py` to set your target subreddit
5. Run the script: `python reddit_subreddit_monitor.py`

## Output

One CSV file, one row per comment, with metadata and boolean tag flags
for filtering by topic.

## Notes

This is a personal learning project. Read-only API access only — no
posting, voting, or interaction with the platform.
