import os, streamlit as st, pandas as pd, joblib
st.set_page_config(page_title="Debug", layout="centered")
st.title("Debug")
st.write("cwd:", os.getcwd())
st.write("files:", sorted(os.listdir(".")))
local="house_price_model.joblib"; alt="/mnt/data/house_price_model.joblib"
st.write("local exists:", os.path.exists(local))
st.write("alt exists:", os.path.exists(alt))
if os.path.exists(local): st.write("local size:", os.path.getsize(local))
if os.path.exists(alt): st.write("alt size:", os.path.getsize(alt))
MODEL = local if os.path.exists(local) else (alt if os.path.exists(alt) else None)
if MODEL is None:
    st.error("MODEL NOT FOUND")
    st.stop()
try:
    joblib.load(MODEL)
    st.success("MODEL LOADED OK")
except Exception as e:
    st.error("MODEL LOAD ERROR: "+str(e))
    st.stop()
st.write("debug UI ready")
