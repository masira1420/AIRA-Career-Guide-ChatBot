import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from dotenv import load_dotenv
import datetime
from streamlit_extras.stylable_container import stylable_container

# Set page config first
st.set_page_config(
    page_title="AIRA Career Guide",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 15px;
        margin-bottom: 8px;
    }
    .stChatMessage.user {
        background-color: #f0f2f6;
        margin-left: 20%;
    }
    .stChatMessage.assistant {
        background-color: #e3f2fd;
        margin-right: 20%;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
        font-weight: bold;
    }
    .stSlider {
        margin: 10px 0;
    }
    .match-score {
        font-size: 1.2em;
        font-weight: bold;
        color: #2e7d32;
    }
    .subject-rating {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

load_dotenv()

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {'knowledge': {}, 'interest': {}}
if 'assessment_complete' not in st.session_state:
    st.session_state.assessment_complete = False

# Load data from database
try:
    conn = sqlite3.connect('careers.db')
    
    # Load subjects
    subjects = pd.read_sql("SELECT id, name FROM subjects ORDER BY id", conn)
    subject_names = subjects['name'].tolist()
    subject_ids = subjects['id'].tolist()
    subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
    
    # Pre-load career data for better performance
    careers = pd.read_sql("""
        SELECT c.soc_code, c.title, s.name as subject_name, 
               cs.req_knowledge, cs.req_importance
        FROM careers c
        JOIN career_subjects cs ON c.soc_code = cs.soc_code
        JOIN subjects s ON cs.subject_id = s.id
        ORDER BY c.soc_code, s.id
    """, conn)
    
    # Group career data by SOC code for easier access
    career_groups = {}
    for soc_code, group in careers.groupby('soc_code'):
        career_groups[soc_code] = {
            'title': group['title'].iloc[0],
            'subjects': dict(zip(group['subject_name'], 
                               zip(group['req_knowledge'].fillna(0), 
                                   group['req_importance'].fillna(0))))
        }
    
    conn.close()
    
except Exception as e:
    st.error(f"❌ Error loading data from database: {str(e)}")
    st.stop()

# API configuration (OpenAI or Gemini)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
API_URL = "https://api.openai.com/v1/chat/completions" if OPENAI_API_KEY != 'your_openai_api_key' else "https://api.google.com/gemini/v1/chat"  # Placeholder for Gemini
MODEL = "gpt-3.5-turbo" if OPENAI_API_KEY != 'your_openai_api_key' else "gemini-pro"

def calculate_career_match(user_ratings, career_data, subject_to_idx):
    """
    Calculate match score between user ratings and career requirements
    Returns: List of tuples (match_score, career_title, soc_code, top_contributors)
    """
    results = []
    
    # Normalize user ratings
    user_knowledge = np.array([user_ratings['knowledge'].get(sub, 0) for sub in subject_to_idx])
    user_interest = np.array([user_ratings['interest'].get(sub, 0) for sub in subject_to_idx])
    
    # Calculate match for each career
    for soc_code, data in career_data.items():
        req_knowledge = np.zeros(len(subject_to_idx))
        req_importance = np.zeros(len(subject_to_idx))
        
        # Fill in requirements for this career
        for sub, (knowledge, importance) in data['subjects'].items():
            if sub in subject_to_idx:
                idx = subject_to_idx[sub]
                req_knowledge[idx] = knowledge or 0
                req_importance[idx] = importance or 0
        
        # Calculate weighted scores
        knowledge_scores = user_knowledge * req_knowledge * (user_interest / 10)
        importance_weights = req_importance * (user_interest / 10)
        
        # Calculate match score (0-100 scale)
        if np.sum(importance_weights) > 0:
            match_score = (np.sum(knowledge_scores) / np.sum(importance_weights)) * 10
            match_score = max(0, min(100, match_score))  # Clamp to 0-100
        else:
            match_score = 0
        
        # Get top contributing subjects
        contributions = []
        for sub in subject_to_idx:
            if sub in data['subjects']:
                idx = subject_to_idx[sub]
                contrib = knowledge_scores[idx] * importance_weights[idx]
                if contrib > 0:
                    contributions.append((sub, contrib))
        
        # Sort by contribution and take top 3
        contributions.sort(key=lambda x: x[1], reverse=True)
        top_contributors = contributions[:3]
        
        results.append((match_score, data['title'], soc_code, top_contributors))
    
    return results

# Function to call API (OpenAI or Gemini)
def get_career_insight(career_title, match_score):
    """Generate a contextual insight about the career match"""
    if match_score >= 90:
        return f"An excellent match! Your skills and interests align very well with {career_title}."
    elif match_score >= 75:
        return f"A strong match! {career_title} could be a great fit based on your profile."
    elif match_score >= 50:
        return f"A good potential match. Consider exploring {career_title} further to see if it aligns with your goals."
    else:
        return f"While not the strongest match, {career_title} might still be worth exploring based on some of your skills."

def call_api(prompt):
    if OPENAI_API_KEY != 'your_openai_api_key':
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': MODEL,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 150
        }
        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    else:
        headers = {
            'Authorization': f'Bearer {GEMINI_API_KEY}',
            'Content-Type': 'application/json'
        }
        data = {
            'model': MODEL,
            'prompt': {'text': prompt},
            'maxTokens': 150
        }
        try:
            response = requests.post(API_URL, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['candidates'][0]['output']
        except Exception as e:
            return f"Error with Gemini API: {str(e)}"

# Main app layout
st.title("🎯 AI Career Guidance Assistant")
st.caption("Discover careers that match your skills and interests")

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    
    # Initial greeting if no chat history
    if not st.session_state.chat_history:
        current_time = datetime.datetime.now().strftime("%I:%M %p, %A, %B %d, %Y")
        greeting = f"""
        👋 **Welcome to AIRA Career Guide!**  
        
        It's {current_time}  
        
        I'm here to help you discover career paths that match your interests and strengths.  
        
        Let's start by getting to know you better. What's your name?"""
        st.session_state.chat_history.append({'role': 'assistant', 'content': greeting})
        st.rerun()

# Ratings section (shown after user provides name)
if st.session_state.user_name and not st.session_state.assessment_complete:
    with st.expander("📝 Rate Your Skills and Interests", expanded=True):
        st.subheader(f"Hello, {st.session_state.user_name}! Rate Your Skills")
        st.write("Help us understand your strengths and interests to provide better career recommendations.")
        
        with st.form("ratings_form"):
            st.write("**Rate each subject from 1 (Beginner) to 10 (Expert) for knowledge, and 1 (Not Interested) to 10 (Very Interested) for interest.**")
            
            # Create two columns for better layout
            cols = st.columns(2)
            
            for i, sub in enumerate(subject_names):
                with cols[i % 2]:
                    with st.container():
                        st.markdown(f"<div class='subject-rating'><b>{sub}</b>", unsafe_allow_html=True)
                        
                        # Knowledge slider
                        st.session_state.user_ratings['knowledge'][sub] = st.slider(
                            "Knowledge (1-10)", 
                            min_value=1, 
                            max_value=10, 
                            value=5, 
                            key=f"k_{sub}",
                            help=f"Rate your knowledge of {sub} from 1 (Beginner) to 10 (Expert)"
                        )
                        
                        # Interest slider
                        st.session_state.user_ratings['interest'][sub] = st.slider(
                            "Interest (1-10)", 
                            min_value=1, 
                            max_value=10, 
                            value=5, 
                            key=f"i_{sub}",
                            help=f"Rate your interest in {sub} from 1 (Not Interested) to 10 (Very Interested)"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
            
            # Submit button with better styling
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                submitted = st.form_submit_button("🚀 Get Career Recommendations", 
                                                type="primary",
                                                use_container_width=True)
                
            if submitted:
                st.session_state.assessment_complete = True
                prompt = f"✅ Thank you for your ratings, {st.session_state.user_name}! I'm analyzing your responses to find the best career matches for you..."
                st.session_state.chat_history.append({'role': 'assistant', 'content': prompt})
                st.rerun()

# Chat input at the bottom
user_input = st.chat_input("Type your message here...")

# Handle chat logic
if user_input:
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    
    if st.session_state.user_name is None and user_input.strip():
        # Set name and prompt for assessment
        st.session_state.user_name = user_input.strip()
        prompt = f"Nice to meet you, {st.session_state.user_name}! Please rate your knowledge and interest in the subjects on the right panel using the sliders, then submit when done."
    elif st.session_state.user_name is not None:
        # Handle other inputs or post-assessment queries
        if not st.session_state.assessment_complete:
            prompt = f"Please complete your ratings on the right, {st.session_state.user_name}!"
        else:
            if "show results" in user_input.lower() or "recommendations" in user_input.lower():
                prompt = f"Sure, {st.session_state.user_name}! Let me display your career recommendations again."
            else:
                prompt = call_api(user_input)
    
    # Add assistant's response to chat history
    st.session_state.chat_history.append({'role': 'assistant', 'content': prompt})
    
    # Rerun to update the UI
    st.rerun()


# Display career recommendations when assessment is complete
if st.session_state.assessment_complete and 'recommendations' not in st.session_state:
    with st.spinner("🔍 Analyzing your skills and interests to find the best career matches..."):
        # Calculate career matches
        recommendations = calculate_career_match(
            st.session_state.user_ratings,
            career_groups,
            subject_to_idx
        )
        
        # Sort by match score (descending) and take top 10
        recommendations.sort(key=lambda x: x[0], reverse=True)
        st.session_state.recommendations = recommendations[:10]
        
        # Add a message to chat
        prompt = "🎉 Here are your top career recommendations based on your skills and interests!"
        st.session_state.chat_history.append({'role': 'assistant', 'content': prompt})
        st.rerun()

# Display recommendations if available
if 'recommendations' in st.session_state and st.session_state.recommendations:
    st.markdown("---")
    st.header("✨ Your Career Matches")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["List View", "Visualization"])
    
    with tab1:
        # Display each recommendation with details
        for idx, (score, title, soc_code, top_contributors) in enumerate(st.session_state.recommendations, 1):
            with st.expander(f"{idx}. {title} - {score:.1f}% Match", expanded=idx==1):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Progress bar for match score
                    st.progress(score/100, f"Match Score: {score:.1f}%")
                    
                    # Top contributing factors
                    st.markdown("**Why this career matches you:**")
                    for sub, contrib in top_contributors:
                        st.markdown(f"- {sub} (Contribution: {contrib:.1f})")
                
                with col2:
                    st.markdown("#### Quick Links")
                    st.page_link(
                        f"https://www.onetonline.org/link/summary/{soc_code}",
                        label="🔍 View on O*NET",
                        help="Detailed career information on O*NET"
                    )
                    st.page_link(
                        f"https://www.bls.gov/ooh/search?q={title.replace(' ', '+')}",
                        label="📊 BLS Outlook",
                        help="Employment outlook from Bureau of Labor Statistics"
                    )
                
                st.markdown("---")
    
    with tab2:
        # Create a visualization of the top matches
        df_viz = pd.DataFrame([
            {'Career': title, 'Match %': score, 'SOC Code': soc_code}
            for score, title, soc_code, _ in st.session_state.recommendations
        ])
        
        # Create a horizontal bar chart
        fig = px.bar(
            df_viz,
            x='Match %',
            y='Career',
            orientation='h',
            title='Your Top Career Matches',
            color='Match %',
            color_continuous_scale='Viridis',
            text='Match %',
            hover_data=['SOC Code']
        )
        
        # Customize the layout
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            xaxis_title='Match Score (%)',
            yaxis_title='',
            height=500,
            margin=dict(l=50, r=50, t=80, b=50),
            coloraxis_showscale=False
        )
        
        # Add percentage signs to the text
        fig.update_traces(
            texttemplate='%{text:.1f}%',
            textposition='outside'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add some additional insights
        st.markdown("#### Key Insights")
        if len(st.session_state.recommendations) >= 3:
            top = st.session_state.recommendations[0]
            st.markdown(f"- **Best Match:** {top[1]} ({top[0]:.1f}% match) - {get_career_insight(top[1], top[0])}")
            
            # Show common skills among top matches
            all_skills = set()
            for _, _, _, skills in st.session_state.recommendations[:3]:
                all_skills.update([s[0] for s in skills])
            
            if all_skills:
                st.markdown("- **Your Top Skills:** " + ", ".join(f"`{s}`" for s in list(all_skills)[:5]) + "...")
        
        st.info("💡 Tip: Click on a career to see more details and explore related opportunities.")

# Add some spacing before the chat input
st.markdown("""
    <style>
        .stChatFloatingInputContainer {
            padding-top: 20px;
            background: white;
            border-top: 1px solid #f0f2f6;
            position: sticky;
            bottom: 0;
            z-index: 100;
        }
        
        /* Ensure chat messages have proper spacing */
        .stChatMessage {
            margin-bottom: 16px;
            border-radius: 12px;
            padding: 12px 16px;
        }
        
        /* Better styling for the expander headers */
        .stExpanderHeader {
            font-size: 1.1em;
            font-weight: 600;
        }
        
        /* Style for the tabs */
        .stTabs [data-baseweb="tab-list"] {
            margin-bottom: 1rem;
        }
    </style>
    
    <script>
        // Auto-scroll chat to bottom
        var chatContainer = document.querySelectorAll('.stChatMessage')[0]?.parentElement;
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Add smooth scrolling to page anchors
        document.addEventListener('DOMContentLoaded', function() {
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {
                anchor.addEventListener('click', function (e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href');
                    if (targetId === '#') return;
                    const targetElement = document.querySelector(targetId);
                    if (targetElement) {
                        targetElement.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
        });
    </script>
""", unsafe_allow_html=True)

# Add a footer with app information
st.markdown("""
    <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666; font-size: 0.9em;">
        <hr style="margin-bottom: 1rem;">
        <p>💡 <strong>AIRA Career Guide</strong> | Version 1.0 | 
        <a href="#" target="_blank">About</a> | 
        <a href="#" target="_blank">Privacy Policy</a> | 
        <a href="#" target="_blank">Contact</a></p>
    </div>
""", unsafe_allow_html=True)