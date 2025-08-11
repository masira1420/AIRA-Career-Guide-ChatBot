# 🎯 AIRA Career Guide

An **AI-powered career guidance assistant** built with **Streamlit** that helps users discover career paths based on their **skills** and **interests**.  
It uses data from a **SQLite database** and an intelligent matching algorithm to provide **personalized career recommendations**.  
Optionally integrates with **OpenAI GPT** or **Google Gemini** for AI-driven insights.

---

## 🚀 Live Demo
🔗 **[Click here to try the live app](https://aira-career-guide-chatbot.streamlit.app/)**  

---

##  Features

- **Interactive Career Assessment**  
  - Rate your **Knowledge** and **Interest** for each subject (1–10 scale).  
  - Dual-slider interface for easy input.

- **AI Career Matching**  
  - Calculates **match scores** between your profile and career requirements.  
  - Shows **top contributing subjects** for each career.

- **Dynamic Career Recommendations**  
  - **List View**: Detailed breakdown with scores, skill contributions, and quick links to O*NET and BLS.  
  - **Visualization**: Horizontal bar chart of your top career matches using Plotly.

- **AI Chat Assistant**  
  - Ask follow-up questions and get contextual insights about careers.  
  - Powered by **OpenAI GPT** or **Google Gemini**.

- **Beautiful UI**  
  - Custom CSS styling for a clean, modern look.  
  - Responsive layout with chat-based interaction.

---

## 📂 Project Structure

```plaintext
.
├── careers.db          # SQLite database containing subjects & careers
├── .env                # Environment variables (API keys)
├── main.py             # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
---

## 🛠️ Tech Stack
- **Frontend:** Streamlit (for interactive UI)
- **Backend:** FastAPI (for API endpoints & logic)
- **AI/NLP:** OpenAI GPT, spaCy, Rasa, BERT
- **Machine Learning:** KNN, Cosine Similarity, LightFM
- **Database:** MySQL / SQLite
- **Other Tools:** Pandas, Scikit-learn, O*NET datasets

## 📊 Workflow
1. **User Interaction** – Student chats with AIRA.
2. **Interest & Skills Profiling** – AIRA asks targeted questions based on knowledge datasets.
3. **Data Processing** – Responses analyzed with NLP & ML models.
4. **Career Matching** – System matches profile to most relevant careers.
5. **Recommendation Delivery** – AIRA suggests top career paths with explanations.

---

## 🔧 Installation & Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/aira-chatbot.git
   cd aira-chatbot
---
📊 Example Output
Best Match: Data Scientist (92.5% match)
Top skills: Mathematics, Computer Science, Statistics

Visualization:
Horizontal bar chart showing match scores for top careers.





