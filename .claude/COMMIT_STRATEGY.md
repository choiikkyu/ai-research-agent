# Commit Strategy for Model Modifications

## 2-Commit Strategy

When modifying or creating new models based on existing implementations, use the **2-commit strategy** to ensure clear and reviewable PR diffs.

### Why This Matters

When a new model is created by copying an existing model and making modifications, a single commit would show the entire new model as "added" files, making it difficult for reviewers to:
- Identify what actually changed from the reference model
- Understand the specific modifications made
- Review the logic changes effectively

### How It Works

#### Commit 1: Copy with Name Changes Only
- Copy all files from the reference model to the new destination path
- Only change identifiers (class names, model names, file references)
- No logic changes in this commit

**Example:**
```
# Reference: xandr_mtl93 -> New: xandr_mtltest
# Changes in Commit 1:
- xandr_mtl93 -> xandr_mtltest (all occurrences)
- Xandr_Mtl93 -> Xandr_Mtltest (class names)
```

#### Commit 2: Apply Actual Modifications
- Make the actual logic changes (layer structure, hyperparameters, etc.)
- This commit's diff shows exactly what was modified

**Example:**
```python
# Changes in Commit 2 (conf.py):
- 'linear_hidden_units': (512, 256, 128),  # 3 layers
+ 'linear_hidden_units': (256, 128),       # 2 layers
```

### Benefits

1. **Clear Diff Review**: Commit 2's diff shows only the actual changes
2. **Easy Rollback**: Can easily identify and revert specific changes
3. **Better Code Review**: Reviewers can focus on what matters
4. **Audit Trail**: Clear history of what was copied vs. modified

### When to Use

Use the 2-commit strategy when **ALL** conditions are met:
1. Task type is `MODEL_TRAINING`
2. There's a reference model being used as base
3. The reference path contains model indicators (`dsp_models`, `vodka`, `whisky`)
4. The reference model has 3+ files

**NOT used for:**
- `FEATURE_ENGINEERING` tasks
- New model implementations without a reference
- Small reference implementations (<3 files)

### Implementation Reference

**Strategy Decision Logic:**
- `src/mcp/tools/code_generator.py:_determine_pr_strategy()` - Determines when to use 2-commit strategy
- `src/mcp/tools/code_generator.py:PRStrategy` - Enum defining available strategies

**PR Creation:**
- `src/mcp/tools/github.py:create_model_pr_with_2commit_strategy()` - 2-commit PR creation
- `src/mcp/tools/github.py:manage_pr()` - Standard single-commit PR creation

**Generated Implementation Fields:**
```python
implementation = {
    ...
    "pr_strategy": "model_modification",  # or "new_implementation"
    "reference_path": "src/.../xandr_mtl93",
    "reference_name": "xandr_mtl93",
}
```

### Commit Message Convention

```
# Commit 1
[AI Agent] Copy {reference_model} -> {new_model} (name changes only)

# Commit 2
[AI Agent] Apply modifications to {new_model}

- Change 1 description
- Change 2 description
```
