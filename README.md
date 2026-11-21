# Adaptive Student Dashboard

## Overview
A small Streamlit prototype demonstrating an AI-related adaptive user interface for a student learning dashboard.

## Run locally
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment (recommended).
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.

## Adaptive behavior
- The user selects a learning focus in the sidebar.
- Opening a resource adds its topic to the in-session preference profile.
- TF-IDF text features and cosine similarity reorder resources based on the profile.
- Detailed/compact mode changes the amount of description shown.
- High-contrast mode changes presentation for readability.
- Reset session learning clears the interaction history.

## Design and limitations
This is a prototype with a small, hand-authored resource catalogue. It uses content-based similarity, not a trained predictive model or generative AI. The preference profile exists only in the current Streamlit session and is not stored in a database. Recommendations are illustrative and should not be treated as evidence of improved learning outcomes.

## Source-code citation
Veeravarapu, S. (2026). *Adaptive Student Dashboard* [Computer software]. GitHub repository: https://github.com/shruthikaveeravarapu-arch/adaptive-student-dashboard
