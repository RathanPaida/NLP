"""Add a 'print segmented output' cell to greedy_segmentation.ipynb."""
import json

notebook_path = 'greedy_segmentation.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Remove any previously added "5.1 Segmented Output" cells to avoid duplicates
nb['cells'] = [
    c for c in nb['cells']
    if '5.1 Segmented Output' not in ''.join(c.get('source', []))
       and 'SEGMENTED OUTPUT' not in ''.join(c.get('source', []))
]

# Find the index of the "## 6. Detailed Analysis" cell
insert_idx = None
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if '## 6. Detailed Analysis' in src:
        insert_idx = i
        break

if insert_idx is None:
    print("Could not find '## 6. Detailed Analysis' — appending at end instead.")
    insert_idx = len(nb['cells'])

md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5.1 Segmented Output for All Test Cases"
    ]
}

code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Print segmented output for all test cases\n",
        "print('=' * 80)\n",
        "print('   SEGMENTED OUTPUT — GREEDY (LONGEST MATCH) — ALL TEST CASES')\n",
        "print('=' * 80)\n",
        "for r in results:\n",
        "    status = 'CORRECT' if r['correct'] else 'WRONG'\n",
        "    print(f\"\\n[{status}] Test Case {r['index'] + 1}\")\n",
        "    print(f\"  Input       : {r['input']}\")\n",
        "    print(f\"  Ground Truth: {r['ground_truth']}\")\n",
        "    print(f\"  Prediction  : {r['prediction']}\")\n",
        "    if not r['correct']:\n",
        "        print(f\"  Edit Dist.  : {r['edit_distance']}\")\n",
        "print('\\n' + '=' * 80)\n",
        "correct_count = sum(1 for r in results if r['correct'])\n",
        "incorrect_count = sum(1 for r in results if not r['correct'])\n",
        "print(f'Total: {len(results)} test cases | Correct: {correct_count} | Incorrect: {incorrect_count}')\n",
        "print('=' * 80)\n"
    ]
}

nb['cells'].insert(insert_idx, code_cell)
nb['cells'].insert(insert_idx, md_cell)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')

print(f"Done! Added 'Segmented Output' cell to {notebook_path}")
print(f"It is now Section 5.1, right before 'Detailed Analysis'.")
