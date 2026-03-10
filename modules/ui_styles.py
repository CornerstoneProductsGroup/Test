import streamlit as st

def apply_global_styles():
    st.markdown("""
    <style>
    .kpi-card{border:1px solid rgba(128,128,128,0.22);border-radius:14px;padding:14px 14px;background: var(--secondary-background-color);}
    .kpi-title{font-size:12px;font-weight:600;letter-spacing:0.02em;color: var(--text-color);opacity: 0.70;}
    .kpi-value{font-size:28px;font-weight:800;line-height:1.15;color: var(--text-color);}
    .kpi-delta{font-size:13px;margin-top:6px;color: var(--text-color);opacity: 0.80;}
    .kpi-delta .delta-abs{ font-weight:800; }
    .kpi-delta .delta-pct{ font-weight:700; opacity:0.88; margin-left:6px; }
    .kpi-delta .delta-note{ opacity:0.75; margin-left:6px; }
    .kpi-big-main{font-size:30px;font-weight:800;line-height:1.05;margin-top:4px;}
    .kpi-big-name{font-size:22px;font-weight:700;line-height:1.15;margin-top:6px;color: var(--text-color);}
    .kpi-big-total{font-size:13px;opacity:0.78;margin-top:6px;color: var(--text-color);}
    .kpi-big-pct{font-size:13px;font-weight:700;margin-top:4px;}
    .intel-card{border:1px solid rgba(128,128,128,0.22);border-radius:16px;padding:14px 16px;background: var(--secondary-background-color);margin-bottom:14px;}
    .intel-header{font-size:12px;font-weight:800;letter-spacing:0.06em;color: var(--text-color);opacity:0.70;}
    .intel-body{margin-top:8px;color: var(--text-color);font-size:15px;line-height:1.45;}
    .intel-body ul{margin: 0;padding-left: 18px;}
    .intel-body li{margin: 6px 0;}
    .report-table{width:100% !important;table-layout:auto;border-collapse: collapse;font-size:14px !important;line-height:1.3;}
    .report-table th, .report-table td{padding:6px 8px;border-bottom:1px solid rgba(128,128,128,0.18);text-align:left;white-space:nowrap;}
    .report-table th{font-size:13px !important;font-weight:700;color:var(--text-color);opacity:0.82;}
    .report-table td{color:var(--text-color);}
    div[data-testid="stDataFrame"] * {font-size:14px !important;}
    </style>
    """, unsafe_allow_html=True)
