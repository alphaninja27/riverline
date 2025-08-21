**Challenge 1**
Robust, human-like debt collection via phone—AI-powered, compliant, and analytics-ready.

**🚀 Features**
Automated Phone Calls: Uses Twilio for real outbound calling (U.S. and global support).

Realistic Voice: Human-like conversational flow (powered by ElevenLabs).

Compliant Script: FDCPA/TCPA-compliant, polite debt collection.

LiveKit Room & Recording: Real-time call room, secure egress recording for audit.

Risk & Sentiment Analytics: Transcripts, user engagement, and risk scoring using ML.

Easy CLI Operation: Run in test or production mode with one file.

Edge Case Handling: Graceful with interruptions, unexpected replies, noise, and compliance requests.

Full Logging & Reporting: SQLite database for storage; structured reports for every call.

**🛠️ Setup**
Clone & Install:

bash
pip install twilio scikit-learn pandas livekit-api livekit-rtc python-dotenv aiohttp
.env File:

text
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
ELEVENLABS_API_KEY=...
DEEPGRAM_API_KEY=...
TWILIO_ACCOUNT_SID=...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1YOURTWILIO
TO_PHONE_NUMBER=+1TARGETNUMBER
Run in Simulation (No Real Calls):

bash
python agent.py test
Make a Real Call:

bash
python agent.py call +1TARGETNUMBER
**📈 Outputs**
Live call recording (mp4, via LiveKit)

Transcript, sentiment & risk analysis (SQLite and JSON reports)

Real-time call status and full log files


# Challenge 2

A production-ready system for **automated testing, evaluation, and self-improvement of debt collection voice agents**. Generate diverse customer personas, simulate complex real-world conversations, track key voicebot metrics, and iteratively optimize your agent — all with one platform.

---

## 🚀 Features

- **Automated Persona Generation:** Create 50+ realistic debtor personalities with diverse traits, stress levels, and payment histories.
- **Conversation Simulation:** Run multi-turn, scenario-specific debt collection call simulations automatically.
- **Comprehensive Metrics:** Evaluate agents using Repetition, Negotiation, Relevance, Compliance, Empathy, Resolution Rate, Satisfaction, and Overall Score.
- **Self-Correcting System:** Pinpoint weaknesses, auto-improve prompts, and repeat until you hit your target score—all hands-free.
- **Intuitive Web App:** Use a Streamlit dashboard to generate personas, test & optimize your agent, and visualize results.
- **Extensible:** Modular Python design, easy to adapt for other industries or integrate with voice APIs.

---

## 🛠️ Quickstart

pip install -r requirements.txt
streamlit run app.py

(See results at http://localhost:8501)
text
or, for CLI demo:
python demo.py

text

---

## 🔍 How It Works

1. **Generate Diverse Personas**: Automatically create customers with realistic psychology and debt traits.
2. **Simulate Calls**: Test your voice agent prompt with dozens of personas across 8 real debt scenarios.
3. **Analyze Performance**: Instantly see which personas and situations your agent struggles with.
4. **Self-Improve**: The platform rewrites your prompt based on test results and measures improvement—no human-in-the-loop needed.
5. **Repeat**: Each iteration gets your agent closer to industry-leading performance and compliance.

---

## 📊 Example Metrics (After Self-Correction)

| Metric               | Before | After  |
|----------------------|--------|--------|
| Overall Score        | 0.56   | 0.78   |
| Negotiation          | 0.51   | 0.73   |
| Compliance           | 0.72   | 0.89   |
| Resolution Rate      | 0.40   | 0.65   |



🛡️ Compliance
All conversations and prompts are built for full U.S. debt collection law and privacy compliance (FDCPA, TCPA, with opt-out/cease support).
