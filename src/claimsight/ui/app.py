"""Streamlit workbench: triage a claim and pick the reserve confidence."""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("CLAIMSIGHT_API_URL", "http://localhost:8250")

st.set_page_config(page_title="claimsight", page_icon="📋", layout="wide")
st.title("📋 claimsight")
st.caption("Severity, a reserve you choose the confidence of, auditable fraud flags, routing")


def _ok() -> bool:
    try:
        return httpx.get(f"{API_URL}/health", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


if not _ok():
    st.error(f"API not reachable at {API_URL}. Start it with `make api`.")
    st.stop()

models = httpx.get(f"{API_URL}/models", timeout=30)
if models.status_code != 200:
    st.error(models.json().get("detail", models.text))
    st.stop()
catalog = models.json()
levels = catalog["reserve_levels"]
st.sidebar.subheader("Model registry")
for version in catalog["versions"]:
    marker = "**Production**" if version["stage"] == "Production" else version["stage"]
    st.sidebar.markdown(
        f"v{version['version']} · {marker}  \n"
        f"median APE {version['metrics'].get('median_ape', float('nan')):.1%} · "
        f"coverage {version['metrics'].get('reserve_coverage', float('nan')):.1%}"
    )
st.sidebar.caption("Only versions passing the reserve-adequacy gate reach Production.")

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

confidence = st.select_slider(
    "Reserve confidence — share of claims the booked reserve should cover",
    options=levels,
    value=0.75 if 0.75 in levels else levels[len(levels) // 2],
    format_func=lambda v: f"{v:.0%}",
)

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
            "reserve_confidence": confidence,
        },
        timeout=30,
    )
    if r.status_code != 200:
        st.error(r.json().get("detail", r.text))
    else:
        body = r.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted severity", f"${body['predicted_severity_usd']:,.0f}")
        c2.metric(
            f"Reserve at {body['reserve_confidence']:.0%}",
            f"${body['suggested_reserve_usd']:,.0f}",
            delta=f"{body['suggested_reserve_usd'] / body['predicted_severity_usd'] - 1:+.1%}",
        )
        c3.metric("Route to", body["route_to"])
        st.caption(
            f"Held-out claims this uplift actually covered: "
            f"**{body['measured_coverage']:.1%}** (asked for {body['reserve_confidence']:.0%}) "
            f"· model v{body['model_version']}"
        )
        seg = body.get("segment_coverage")
        if seg:
            st.caption(
                f"On {seg['segment']} specifically, the same global uplift covered "
                f"{seg['coverage']:.1%} of {seg['n_test']} held-out claims."
            )
        if body["fraud_flags"]:
            st.subheader("Fraud flags")
            for f in body["fraud_flags"]:
                st.markdown(f"🚩 **{f['flag']}** — {f['evidence']}")
        else:
            st.success("No fraud flags.")

st.divider()
rq = httpx.get(f"{API_URL}/queue-stats", params={"reserve_confidence": confidence}, timeout=120)
if rq.status_code == 200:
    body = rq.json()
    st.subheader(f"Book routing and capital ({body['sampled_claims']:,}-claim sample)")
    st.bar_chart(body["queue_depth"])
    st.caption(
        f"SIU share: {body['siu_share']:.1%} · weekly capacity: {body['weekly_capacity']} · "
        f"reserves booked at {body['reserve_confidence']:.0%}: "
        f"${body['reserves_booked_usd']:,.0f} against ${body['claimed_total_usd']:,.0f} claimed"
    )
