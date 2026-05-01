# MaxSup Logit Simulator — Booth Demo

Interactive demo for the poster *MaxSup: Overcoming Representation Collapse in Label Smoothing*.

## Run (one command)

```bash
conda run -n 7606 streamlit run demo/app.py
```

Then open **http://localhost:8501** in a browser.

## What it shows

- Three sliders set logit values for `cat / fox / dog`
- Choose the **GT (ground truth) class** from the dropdown
- Two live bar charts show which logit each method **suppresses** (downward arrow)
- A one-step section shows what happens to each logit after a gradient update
- When the model is **wrong** (fox > cat, GT = cat), a callout highlights:
  - LS widens the gap → **error amplified**
  - MaxSup narrows the gap → **error corrected**

## Key formula

```
L_LS     = α ( z_gt  − mean(z) )   ← suppresses the labeled class
L_MaxSup = α ( z_max − mean(z) )   ← suppresses the current top-1

One symbol:  z_gt  →  z_max
```

## Booth talking points

| Audience question | Answer |
|---|---|
| "What does MaxSup do differently?" | Point to the two arrows — LS arrow follows GT, MaxSup arrow follows the wrong top-1 |
| "Is that a big change?" | Drag cat slider above fox → both arrows jump to cat → same behaviour on correct samples |
| "Why does it help?" | Click "After 1 step" — LS gap widens, MaxSup gap narrows |
| "Is it expensive?" | "No architecture change, virtually no extra compute" |
