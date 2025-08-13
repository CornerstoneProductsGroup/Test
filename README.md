# Retail Order Splitter (HD + Lowe's)

- **Home Depot**: extracts the value directly under the **'Model Number'** label
- **Lowe's**: extracts the value directly under the **'Model #'** (or **'Model Number'**) label
- Match with a **vendor map (.xlsx)** to route pages into per-vendor PDFs
- **Print Pack**: combine selected vendors in strict, alphabetical order (exact list)
- **Review & Fix** UI to resolve uncertain pages
- Sidebar: per-retailer downloads + default map managers

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
