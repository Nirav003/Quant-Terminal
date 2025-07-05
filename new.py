from flask import Blueprint, render_template
import feedparser

news_bp = Blueprint('news_bp', __name__)

# Yahoo Finance RSS feed (Top Stories)
RSS_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^DJI,^GSPC,^IXIC&region=US&lang=en-US"

def get_finance_news():
    feed = feedparser.parse(RSS_FEED)
    if feed.bozo:
        return [f"⚠️ Error parsing RSS: {feed.bozo_exception}"]
    
    titles = [entry.title for entry in feed.entries[:5]]
    return titles if titles else ["No news found."]

@news_bp.route('/news')
def show_news():
    news_titles = get_finance_news()
    return render_template('news.html', news_titles=news_titles)
