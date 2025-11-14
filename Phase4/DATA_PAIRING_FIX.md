# Data Pairing Fix

## Problem
Line 74 uses index-based pairing: `text = self.text_df.iloc[idx]['text']`  
This fails if CSV and image dataset have different orders/lengths.

## Solution

### Step 1: Add Translation Dictionary (if labels differ)

In `main()`, after loading CSV (around line 84):

```python
df = pd.read_csv(TEXT_CSV)

# If text labels ≠ image labels, add translation:
translation_dictionary = {
    0: 5,  # Sadness → Sad
    1: 3,  # Joy → Happy
    2: 3,  # Love → Happy
    3: 0,  # Anger → Angry
    4: 2,  # Fear → Fear
    5: 6   # Surprise → Surprise
}
df['label'] = df['label'].replace(translation_dictionary)
```

### Step 2: Fix __getitem__ Method

Replace lines 72-76 in `RAFDB_TextFusionDataset`:

```python
def __getitem__(self, idx):
    img, img_label = self.img_data[idx]
    
    # Match by label, not index
    matching_texts = self.text_df[self.text_df['label'] == img_label]['text']
    if len(matching_texts) > 0:
        text = matching_texts.sample(n=1).iloc[0]
    else:
        text = self.text_df.sample(n=1).iloc[0]['text']
    
    text_enc = encode(text, self.vocab, MAX_LEN)
    return img, text_enc, img_label
```

## Notes
- Check if CSV column is named `'label'` (change if different)
- Verify translation dictionary matches your label schemes
- Match by emotion label, not array index
