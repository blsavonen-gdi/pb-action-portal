"""
Literature Stats — user submission form.

Lets visitors propose a new quantitative datapoint. Submissions are appended to
a Google Sheet (reviewed manually before any are folded into
data/literature_datapoints.csv).

Configuration (Streamlit secrets / .streamlit/secrets.toml):

    [gcp_service_account]
    # the full service-account JSON key, field by field
    type = "service_account"
    project_id = "..."
    private_key_id = "..."
    private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
    client_email = "portal-writer@your-project.iam.gserviceaccount.com"
    client_id = "..."
    token_uri = "https://oauth2.googleapis.com/token"

    [literature_submissions]
    sheet_id = "the long id from the sheet URL"
    worksheet = "Submissions"          # optional, defaults to "Submissions"

Share the target Google Sheet with the service account's client_email (Editor).
When the secrets are absent (e.g. local dev), the form still renders and, on
submit, offers the composed row as a CSV download + a pre-filled email link so
nothing is lost.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

_CONTACT_EMAIL = "ben.savonen@globaldevincubator.org"

# Column order written to the sheet (first row is a matching header).
_HEADERS = [
    "submitted_utc", "submitter_name", "submitter_email",
    "statistic", "value", "units", "geography", "year",
    "topic", "evidence_type", "source", "page", "quote", "notes",
    "status",
]

# Kept local (not imported from app.py) to avoid a circular import.
_TOPIC_OPTIONS = [
    "Collection & ULAB volumes", "Recycling efficiency", "Recycling capacity & scale",
    "Recycling economics & cost", "Battery composition & weight", "Battery lifespan",
    "Lead demand & consumption", "Trade flows", "Health & environmental exposure",
    "Off-grid solar & mini-grids", "Other / not sure",
]
_EVIDENCE_OPTIONS = [
    "Measured / empirical", "Survey", "Government / agency statistic",
    "Industry report", "Peer-reviewed study", "Modeled / estimated",
    "Expert judgement", "Other / not sure",
]


def _submissions_configured() -> bool:
    """True only when both the service account and the target sheet id are set."""
    try:
        sa = st.secrets["gcp_service_account"]
        sid = st.secrets["literature_submissions"]["sheet_id"]
    except Exception:
        return False
    return bool(sa) and bool(sid)


@st.cache_resource(show_spinner=False)
def _get_worksheet():
    """Authorise with the service account and return the target worksheet,
    creating it (with a header row) if it does not exist yet."""
    import gspread
    from google.oauth2.service_account import Credentials

    sa_info = dict(st.secrets["gcp_service_account"])
    conf = st.secrets["literature_submissions"]
    sheet_id = conf["sheet_id"]
    ws_name = conf.get("worksheet", "Submissions")

    creds = Credentials.from_service_account_info(
        sa_info, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(ws_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=ws_name, rows=1000, cols=len(_HEADERS))
    # Ensure a header row exists.
    if not ws.row_values(1):
        ws.update("A1", [_HEADERS])
    return ws


def _row_from(payload: dict) -> list[str]:
    return [str(payload.get(h, "")) for h in _HEADERS]


def render_submission_form() -> None:
    st.divider()
    with st.expander("➕ Submit a new statistic"):
        st.caption(
            "Know a well-sourced figure that belongs here? Share it below. "
            "Submissions are reviewed before being added — please include the "
            "source so we can verify it."
        )

        configured = _submissions_configured()
        if not configured:
            st.info(
                "Live submission isn't configured on this instance yet. You can "
                "still fill in the form — on submit you'll get a copy to download "
                "and an email link so nothing is lost.",
                icon="ℹ️",
            )

        with st.form("lit_submit_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            statistic = c1.text_input(
                "Statistic *", placeholder="e.g. ULAB collection rate, Kenya",
            )
            value = c2.text_input(
                "Value *", placeholder="e.g. 62% or 1.2–1.5 million tonnes",
            )

            c3, c4, c5 = st.columns(3)
            units = c3.text_input("Units", placeholder="e.g. %, t Pb, kg/battery")
            geography = c4.text_input("Geography", placeholder="Country / region")
            year = c5.text_input("Year", placeholder="e.g. 2023")

            c6, c7 = st.columns(2)
            topic = c6.selectbox("Topic", _TOPIC_OPTIONS, index=len(_TOPIC_OPTIONS) - 1)
            evidence = c7.selectbox(
                "Evidence type", _EVIDENCE_OPTIONS, index=len(_EVIDENCE_OPTIONS) - 1
            )

            source = st.text_input(
                "Source / citation *",
                placeholder="Author (year), title, publisher — or a URL/DOI",
            )
            c8, _ = st.columns([1, 3])
            page = c8.text_input("Page / location", placeholder="e.g. p.34")
            quote = st.text_area(
                "Supporting quote or context",
                placeholder="Paste the sentence from the source that states this figure.",
            )
            notes = st.text_area("Notes (optional)")

            st.markdown("**Your details** (optional — so we can follow up)")
            c9, c10 = st.columns(2)
            name = c9.text_input("Name")
            email = c10.text_input("Email")

            submitted = st.form_submit_button("Submit statistic", type="primary")

        if not submitted:
            return

        # ---- validation ------------------------------------------------
        missing = [
            lbl for lbl, val in
            [("Statistic", statistic), ("Value", value), ("Source / citation", source)]
            if not val.strip()
        ]
        if missing:
            st.error("Please fill in the required field(s): " + ", ".join(missing) + ".")
            return

        payload = {
            "submitted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "submitter_name": name.strip(),
            "submitter_email": email.strip(),
            "statistic": statistic.strip(),
            "value": value.strip(),
            "units": units.strip(),
            "geography": geography.strip(),
            "year": year.strip(),
            "topic": topic,
            "evidence_type": evidence,
            "source": source.strip(),
            "page": page.strip(),
            "quote": quote.strip(),
            "notes": notes.strip(),
            "status": "pending",
        }

        if configured:
            try:
                ws = _get_worksheet()
                ws.append_row(_row_from(payload), value_input_option="USER_ENTERED")
                st.success(
                    "Thanks! Your statistic has been submitted for review. "
                    "We'll verify the source before adding it."
                )
                return
            except Exception as exc:  # fall through to the offline path
                st.warning(
                    "Couldn't reach the submissions sheet, so here's a copy to send "
                    f"us directly instead. (Details: {exc})"
                )

        # ---- offline / fallback path -----------------------------------
        _offline_fallback(payload)


def _offline_fallback(payload: dict) -> None:
    row_csv = pd.DataFrame([payload]).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download your submission (CSV)",
        row_csv,
        file_name="pb_action_literature_submission.csv",
        mime="text/csv",
    )
    from urllib.parse import quote as _q

    body = "\n".join(f"{h}: {payload.get(h, '')}" for h in _HEADERS if payload.get(h))
    subject = f"Pb Action Portal — new literature statistic: {payload['statistic']}"
    mailto = f"mailto:{_CONTACT_EMAIL}?subject={_q(subject)}&body={_q(body)}"
    st.markdown(f"[✉ Email this submission to the Pb Action team]({mailto})")
