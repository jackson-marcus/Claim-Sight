"""Streamlit demo: triage a claim, see reserve + flags + routing."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("CLAIMSIGHT_API_URL", "http://localhost:8250")

st.set_page_config(page_title="claimsight", page_icon="📋", layout="wide")
st.title("📋 claimsight")
st.caption("Severity + reserve suggestion, auditable fraud flags, capacity-aware routing")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    ctype = st.selectbox(
        "Claim type",
        ["auto_collision", "auto_theft", "property_water", "property_fire", "liability"],
    )
    amount = st.number_input("Claimed amount ($)", 100.0, 500000.0, 12000.0)
    injuries = st.number_input("Injuries", 0, 20, 0)
    police = st.checkbox("Police report filed", value=True)
with col2:
    delay = st.slider("Report delay (days)", 0, 120, 3)
    tenure = st.slider("Policy tenure (days)", 1, 3000, 400)
    priors = st.number_input("Prior claims (3y)", 0, 30, 0)
    vage = st.slider("Vehicle age", 0, 20, 6)

if st.button("Triage claim", type="primary"):
    r = httpx.post(
        f"{API_URL}/triage",
        json={
            "claim_type": ctype,
            "vehicle_age": int(vage),
            "injuries": int(injuries),
            "police_report": int(police),
            "report_delay_days": int(delay),
            "policy_tenure_days": int(tenure),
            "prior_claims_3y": int(priors),
            "claimed_amount": amount,
        },
        timeout=30,
    )
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
    else:
        body = r.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted severity", f"${body['predicted_severity_usd']:,.0f}")
        c2.metric("Suggested reserve", f"${body['suggested_reserve_usd']:,.0f}")
        c3.metric("Route to", body["route_to"])
        if body["fraud_flags"]:
            st.subheader("Fraud flags")
            for f in body["fraud_flags"]:
                st.markdown(f"🚩 **{f['flag']}** — {f['evidence']}")
        else:
            st.success("No fraud flags.")

st.divider()
rq = httpx.get(f"{API_URL}/queue-stats", timeout=120)
if rq.status_code == 200:
    body = rq.json()
    st.subheader("Book routing (2,000-claim sample)")
    st.bar_chart(body["queue_depth"])
    st.caption(f"SIU share: {body['siu_share']:.1%} · weekly capacity: {body['weekly_capacity']}")
