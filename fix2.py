c = open('football_bot_FINAL.py', 'r', encoding='utf-8').read()
old = "parse_mode='Markdown'"
n = c.count(old)
c = c.replace(old, '')
# Clean up trailing commas + whitespace before closing paren
import re
c = re.sub(r',\s*\)', ')', c)
open('football_bot_FINAL.py', 'w', encoding='utf-8').write(c)
print(f"Removed {n} remaining parse_mode instances")
