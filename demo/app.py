"""Local / Space live demo: upload an mp4, run Part A, show a timeline.

    streamlit run demo/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution import detect_events  # noqa: E402

st.set_page_config(page_title="PDPU traffic events", layout="wide")
st.title("PDPU — traffic event demo")
st.caption("Upload an .mp4 (about 2 minutes is enough). CPU is fine; a short clip is kinder.")

uploaded = st.file_uploader("Video", type=["mp4"])
if uploaded is None:
    st.info("Waiting for a file.")
    st.stop()

tmp = Path(st.session_state.get("tmp_path") or "")
cache_key = f"{uploaded.name}-{uploaded.size}"
if st.session_state.get("cache_key") != cache_key:
    dest = Path("/tmp") / f"pdpu_{uploaded.name}"
    dest.write_bytes(uploaded.getvalue())
    st.session_state["tmp_path"] = str(dest)
    st.session_state["cache_key"] = cache_key
    st.session_state.pop("events", None)

path = st.session_state["tmp_path"]
st.video(path)

if st.button("Detect events") or "events" in st.session_state:
    if "events" not in st.session_state:
        with st.spinner("Running detector + tracker + rules…"):
            st.session_state["events"] = detect_events(path)
    events = st.session_state["events"]
    st.metric("Events", len(events))
    if not events:
        st.warning("No events (or the model is still a conservative baseline).")
    else:
        import pandas as pd

        df = pd.DataFrame(events, columns=["start_sec", "end_sec", "label"])
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df["label"].value_counts())
