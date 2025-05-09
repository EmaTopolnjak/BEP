import json
from collections import Counter

# Load JSON file
with open("data/rotations_IHC/image_rotations_IHC.json", "r") as f:
    data = json.load(f)

# Total number of keys
total_keys = len(data)

# Count skipped entries and reasons
skipped_count = 0
skipped_reasons = Counter()

for key, value in data.items():
    if isinstance(value, dict) and "skipped" in value:
        skipped_count += 1
        reason = value["skipped"]
        skipped_reasons[reason] += 1

# Output results
print(f"Number images before rotating: {total_keys}")
print(f"Number of skipped images: {skipped_count}")
print("Skipped reasons breakdown:")
for reason, count in skipped_reasons.items():
    print(f"  {reason}: {count}")