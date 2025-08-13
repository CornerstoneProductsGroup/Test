# Home Depot Order Splitter (Anchor-Based)

- Extracts **Model** directly under the **'Model Number'** label (Home Depot slips)
- Matches to a built-in or uploaded **.xlsx** Vendor Map
- Auto-splits pages into per-vendor PDFs
- **Print Pack**: combines selected vendors into one PDF for one-tap printing
- **Review & Fix**: dropdowns to resolve low-confidence pages
- Sidebar keeps **current ZIP**, **Print Pack**, and **previous batches**

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
