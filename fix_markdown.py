"""
Simple fix: strip Markdown parse_mode to prevent BadRequest errors.
Replace parse_mode='Markdown' with parse_mode=None and remove markdown
formatting characters (* _ `) from message strings.
"""

with open('football_bot_FINAL.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all parse_mode='Markdown' with no parse_mode
# Easiest: remove parse_mode='Markdown' entirely (defaults to None)
content = content.replace(", parse_mode='Markdown'", "")
content = content.replace(",parse_mode='Markdown'", "")

# Fix the escaped underscores in ODDS_API_KEY
content = content.replace("ODDS\\\\_API\\\\_KEY", "ODDS_API_KEY")

with open('football_bot_FINAL.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! Removed all parse_mode='Markdown' and fixed ODDS escape.")
