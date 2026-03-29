import pandas as pd
import random

# Mapping real merchants to categories
data_map = {
    "Dining": ["Starbucks", "McDonalds", "Chipotle", "Dunkin", "Subway", "Burger King", "Panda Express"],
    "Transport": ["Uber", "Lyft", "Chevron", "Shell", "Exxon", "Gas", "Tesla Supercharger"],
    "Groceries": ["Trader Joes", "Whole Foods", "Safeway", "Costco", "Target", "Walmart", "Kroger"],
    "Services": ["Netflix", "Spotify", "Hulu", "Adobe", "AWS", "Google Cloud", "iCloud"]
}

rows = []
for _ in range(300): # 300 examples is plenty for a "Smart" V1
    cat = random.choice(list(data_map.keys()))
    merchant = random.choice(data_map[cat])
    # Randomly add suffixes like "Inc" or "Store #123" to make it look like a real bank statement
    suffix = random.choice(["", "Inc", "Store #101", "Payment", "Purchase", "Mobile"])
    rows.append({"description": f"{merchant} {suffix}".strip(), "category": cat})

df = pd.DataFrame(rows)
df.to_csv("app/ml/transactions.csv", index=False)
print("✅ Created app/ml/transactions.csv with 300 examples.")
