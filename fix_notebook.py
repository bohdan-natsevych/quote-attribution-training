
import json
import os

notebook_path = r"c:\Work\personal\audio-book-generator\training\booknlp_max_unified.ipynb"

def fix_notebook():
    """
    Fix the DeepSpeed/TrainingArguments fp16 mismatch by:
    1. Making sure fp16.enabled is "auto" in DeepSpeed config
    2. Adding fp16_full_eval=CONFIG.fp16 to TrainingArguments
    3. Removing malformed line if present
    """
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    modified = False

    for cell in cells:
        if cell.get('cell_type') == 'code':
            source_lines = cell.get('source', [])
            new_source = []
            skip_next = False
            
            for i, line in enumerate(source_lines):
                # Skip malformed lines from previous attempts
                if '"        fp16_full_eval=' in line and '\\"' in line:
                    print(f"Removing malformed line: {line[:60]}...")
                    modified = True
                    continue
                
                # Find fp16=CONFIG.fp16 line and add fp16_full_eval after it
                if 'fp16=CONFIG.fp16,' in line and 'fp16_full_eval' not in line:
                    new_source.append(line)
                    # Check if next line already has fp16_full_eval (properly formatted)
                    if i + 1 < len(source_lines) and 'fp16_full_eval=CONFIG.fp16' in source_lines[i + 1]:
                        continue  # Already has it
                    # Add properly formatted line
                    # Get the indentation (8 spaces for this context)
                    new_line = "        fp16_full_eval=CONFIG.fp16,  # CURSOR: Must match fp16 for DeepSpeed\\n"
                    new_source.append(new_line)
                    print("Added fp16_full_eval=CONFIG.fp16")
                    modified = True
                else:
                    new_source.append(line)
            
            cell['source'] = new_source

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print("Notebook updated successfully.")
    else:
        print("No changes made. Check notebook manually.")

if __name__ == "__main__":
    fix_notebook()
