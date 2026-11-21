import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Adaptive Student Dashboard", page_icon="🎓", layout="wide")

RESOURCES = pd.DataFrame([
    {"title":"Getting started with Python", "topic":"Programming", "level":"Beginner",
     "description":"Practice variables, conditions, loops, and functions with short examples.",
     "tags":"python coding programming beginner syntax"},
    {"title":"Understanding machine learning", "topic":"Machine Learning", "level":"Intermediate",
     "description":"Explore supervised learning, model training, and evaluation.",
     "tags":"machine learning models classification regression data"},
    {"title":"Natural language processing basics", "topic":"NLP", "level":"Intermediate",
     "description":"Learn tokenization, text features, embeddings, and language models.",
     "tags":"nlp language text tokens embeddings transformers"},
    {"title":"Responsible and ethical AI", "topic":"AI Ethics", "level":"Beginner",
     "description":"Review fairness, transparency, privacy, accountability, and oversight.",
     "tags":"ethics fairness privacy transparency responsible ai"},
    {"title":"Building accessible interfaces", "topic":"HCI", "level":"Beginner",
     "description":"Apply readable typography, keyboard access, contrast, and clear navigation.",
     "tags":"hci accessibility usability interface design contrast"},
    {"title":"Data visualization essentials", "topic":"Data Science", "level":"Beginner",
     "description":"Choose charts and communicate patterns clearly with data.",
     "tags":"data science visualization charts analysis"},
    {"title":"Neural networks in practice", "topic":"Deep Learning", "level":"Advanced",
     "description":"Study neural network layers, optimization, and model behavior.",
     "tags":"deep learning neural networks training optimization"},
    {"title":"Prompting and generative AI", "topic":"Generative AI", "level":"Intermediate",
     "description":"Practice writing prompts and evaluating generated responses.",
     "tags":"generative ai prompts llm language models"}
])

st.title("🎓 Adaptive Student Dashboard")
st.caption("A small demonstration of an interface that learns from your selected interests and interactions.")

if "interest" not in st.session_state:
    st.session_state.interest = "Machine Learning"
if "history" not in st.session_state:
    st.session_state.history = []
if "view" not in st.session_state:
    st.session_state.view = "Detailed"
if "accessible" not in st.session_state:
    st.session_state.accessible = False

with st.sidebar:
    st.header("Your preferences")
    st.selectbox("Current learning focus", sorted(RESOURCES["topic"].unique()),
                 key="interest", help="Changing this updates the recommendations.")
    st.radio("Information density", ["Detailed", "Compact"], key="view",
             help="The dashboard changes how much information each resource card displays.")
    st.toggle("High-contrast reading mode", key="accessible")
    st.caption("The app uses your choices and resource clicks only within this session.")

# Build a lightweight content profile from the selected topic and clicked topics.
profile_topics = [st.session_state.interest] + st.session_state.history
profile_text = " ".join(profile_topics)
resource_text = (RESOURCES["topic"] + " " + RESOURCES["tags"] + " " + RESOURCES["description"]).tolist()
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(resource_text + [profile_text])
scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
ranked = RESOURCES.copy()
ranked["match"] = scores
ranked = ranked.sort_values(["match", "title"], ascending=[False, True])

# A simple interaction model: opening a resource adds its topic to the current user profile.
st.subheader("Recommended learning resources")
st.write(f"Your current profile emphasizes **{st.session_state.interest}**.")
if st.session_state.history:
    st.caption("Topics you explored this session: " + ", ".join(dict.fromkeys(st.session_state.history)))
else:
    st.caption("Open a resource to teach the dashboard more about your interests.")

if st.session_state.accessible:
    st.markdown("""
    <style>
    .stApp {background-color:#111111;color:#ffffff;}
    [data-testid="stMarkdownContainer"] p, label, h1, h2, h3 {color:#ffffff !important;}
    div[data-testid="stVerticalBlockBorderWrapper"] {border:2px solid #ffffff;}
    </style>
    """, unsafe_allow_html=True)
    st.info("High-contrast mode is active. Turn it off in the sidebar to restore the default theme.")

columns = st.columns(2)
for idx, (_, item) in enumerate(ranked.iterrows()):
    with columns[idx % 2]:
        with st.container(border=True):
            st.markdown(f"**{item['title']}**")
            st.caption(f"{item['topic']} · {item['level']} · Match {item['match']:.2f}")
            if st.session_state.view == "Detailed":
                st.write(item["description"])
            else:
                st.caption(item["description"][:78] + ("…" if len(item["description"]) > 78 else ""))
            if st.button("Explore this resource", key=f"open_{item['title']}"):
                st.session_state.history.append(item["topic"])
                st.session_state["last_opened"] = item["title"]
                st.rerun()

if "last_opened" in st.session_state:
    st.success(f"Opened: {st.session_state.last_opened}. Your recommendations have been recalculated.")

st.divider()
st.subheader("How adaptation works")
st.write(
    "The selected focus and topics opened during this session form a small preference profile. "
    "TF-IDF represents the profile and resource descriptions as weighted text features; cosine "
    "similarity ranks the resources. The selected information-density and contrast settings also "
    "change the presentation immediately. This is a classroom prototype, not a production learner model."
)
if st.button("Reset session learning"):
    st.session_state.history = []
    st.session_state.pop("last_opened", None)
    st.rerun()
