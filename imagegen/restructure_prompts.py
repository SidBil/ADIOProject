#!/usr/bin/env python3
"""
Restructure prompts.jsonl so each prompt is a separate entry with structure_word field.
"""

import json

# Structure words to extract
STRUCTURE_WORDS = ["what", "size", "where", "when", "color", "mood", "number", "movement", "sound", "perspective"]

# Since the original file had 25 unique image_index values (1-25)
# Create separate prompt entries for each image_index and each structure_word
new_prompts = []
prompt_id = 1

# Create entries for image_index 1-25, each with all structure words
for image_index in range(1, 26):  # 1 to 25
    for structure_word in STRUCTURE_WORDS:
        new_prompts.append({
            "prompt_id": prompt_id,
            "image_index": image_index,
            "structure_word": structure_word,
            "prompt_text": ""  # Empty for now, can be filled in later
        })
        prompt_id += 1

# Write the restructured data
output_file = "prompts.jsonl"
with open(output_file, 'w', encoding='utf-8') as f:
    for prompt in new_prompts:
        f.write(json.dumps(prompt) + '\n')

print(f"Created {len(new_prompts)} prompt entries")
print(f"Structure: {len(range(1, 26))} images × {len(STRUCTURE_WORDS)} structure words = {len(new_prompts)} prompts")
print(f"Output written to {output_file}")
