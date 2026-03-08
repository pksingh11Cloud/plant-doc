import os
import streamlit as st
import base64
import json
from openai import OpenAI
from PIL import Image
import io
from dotenv import load_dotenv

load_dotenv()  # loads OPENAI_API_KEY from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PaudeKaDoctor - Powered by AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0d1f0f 0%, #122614 40%, #0a1a0c 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 700;
    color: #a8e6b0;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem;
    text-shadow: 0 0 40px rgba(168,230,176,0.3);
}
.hero p {
    color: #5a8a62;
    font-size: 1.05rem;
    font-weight: 300;
}

/* ── Upload zone ── */
.upload-hint {
    border: 1.5px dashed #2a5c30;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    color: #4a7a52;
    font-size: 0.9rem;
    margin: 1rem 0;
    background: rgba(255,255,255,0.02);
}

/* ── Cards ── */
.result-card {
    background: rgba(15, 35, 17, 0.85);
    border: 1px solid #1e4d24;
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin: 1rem 0;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.result-card h3 {
    font-family: 'Playfair Display', serif;
    color: #a8e6b0;
    font-size: 1.2rem;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.result-card p, .result-card li {
    color: #b0ccb4;
    line-height: 1.7;
    font-size: 0.95rem;
}
.result-card ul {
    padding-left: 1.2rem;
}
.result-card li {
    margin-bottom: 0.4rem;
}

/* ── Severity badge ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.85rem;
    border-radius: 50px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-low    { background: #1a4d20; color: #6ddc7a; border: 1px solid #2d7a36; }
.badge-medium { background: #4d3a00; color: #f0b429; border: 1px solid #7a5c00; }
.badge-high   { background: #4d0a0a; color: #f06060; border: 1px solid #7a1a1a; }
.badge-healthy{ background: #0d3320; color: #40d080; border: 1px solid #1a6640; }

/* ── Divider ── */
.divider {
    border: none;
    border-top: 1px solid #1a3d1e;
    margin: 1.5rem 0;
}

/* ── Button styling ── */
.stButton > button {
    background: linear-gradient(135deg, #2a7a34 0%, #1e5c26 100%);
    color: #d0f0d4;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 2.5rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 500;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s;
    box-shadow: 0 4px 20px rgba(42,122,52,0.35);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #34943f 0%, #267030 100%);
    box-shadow: 0 6px 28px rgba(42,122,52,0.55);
    transform: translateY(-1px);
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02);
    border: 1.5px dashed #2a5c30;
    border-radius: 16px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"] label {
    color: #5a8a62 !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #2a7a34 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0a1a0c;
    border-right: 1px solid #1a3d1e;
}
[data-testid="stSidebar"] * {
    color: #7aaa82 !important;
}

/* ── Input ── */
.stTextInput input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid #2a5c30 !important;
    border-radius: 10px !important;
    color: #c0e0c4 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Image container ── */
.img-container {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #1e4d24;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: #2a5c30;
    font-size: 0.8rem;
    padding: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: encode image to base64 ─────────────────────────────────────────────
def encode_image(image_file) -> str:
    return base64.b64encode(image_file.read()).decode("utf-8")


# ── Helper: call GPT-4o Vision ─────────────────────────────────────────────────
def analyze_plant(client: OpenAI, b64_image: str, mime_type: str) -> dict:
    prompt = """You are an expert botanist and plant pathologist. Analyze this plant image carefully.

Return ONLY a valid JSON object with this exact structure (no markdown, no explanation):
{
  "plant_name": "common name of the plant (or 'Unknown Plant' if unclear)",
  "health_status": "Healthy" or "Diseased" or "Stressed" or "Pest Infestation",
  "severity": "None" or "Low" or "Medium" or "High",
  "issues_detected": ["issue1", "issue2"],
  "diagnosis": "2-3 sentence detailed diagnosis of what you observe",
  "causes": ["cause1", "cause2", "cause3"],
  "remedies": [
    {"title": "Remedy name", "description": "Detailed actionable remedy description"},
    {"title": "Remedy name", "description": "Detailed actionable remedy description"}
  ],
  "prevention_tips": ["tip1", "tip2", "tip3"],
  "urgency": "Low" or "Medium" or "High",
  "confidence": 85
}"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{b64_image}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        max_tokens=1200,
        temperature=0.2,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


# ── Severity badge HTML ────────────────────────────────────────────────────────
def severity_badge(level: str) -> str:
    cls_map = {
        "none": "badge-healthy",
        "low": "badge-low",
        "medium": "badge-medium",
        "high": "badge-high",
    }
    cls = cls_map.get(level.lower(), "badge-low")
    return f'<span class="badge {cls}">{level}</span>'


# ── Main UI ────────────────────────────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🌿 PaudeKaDoctor</h1>
        <p>Powered by AI &nbsp;·&nbsp; Upload a plant photo &mdash; get instant diagnosis & remedies powered by GPT-4o Vision</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar – info
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        if OPENAI_API_KEY:
            st.success("✅ API key loaded from .env")
        else:
            st.error("❌ OPENAI_API_KEY not found in .env")
        st.markdown("---")
        st.markdown("**Supported formats**")
        st.markdown("JPG · PNG · WEBP")
        st.markdown("**Model**")
        st.markdown("GPT-4o (high detail)")
        st.markdown("---")
        st.markdown("Add your key to a `.env` file:")
        st.code("OPENAI_API_KEY=sk-...", language="bash")

    # Main area
    uploaded_file = st.file_uploader(
        "Upload plant image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )

    if not uploaded_file:
        st.markdown("""
        <div class="upload-hint">
            📷 &nbsp; Drag & drop your plant image above, or click to browse<br>
            <small>Works best with clear, well-lit photos of leaves, stems, or the whole plant</small>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        # Show uploaded image
        img = Image.open(uploaded_file)
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            st.image(img, use_container_width=True, caption="Uploaded plant image")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        analyze_btn = st.button("🔍 Analyze Plant", use_container_width=True)

        if analyze_btn:
            if not OPENAI_API_KEY:
                st.error("⚠️ OPENAI_API_KEY not found. Please add it to your `.env` file and restart the app.")
                return

            with st.spinner("Analyzing your plant with GPT-4o Vision..."):
                try:
                    client = OpenAI(api_key=OPENAI_API_KEY)

                    # Re-read bytes for encoding
                    uploaded_file.seek(0)
                    b64 = encode_image(uploaded_file)
                    mime = f"image/{uploaded_file.type.split('/')[-1]}"

                    result = analyze_plant(client, b64, mime)

                except json.JSONDecodeError:
                    st.error("Could not parse the AI response. Please try again.")
                    return
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    return

            # ── Results ──────────────────────────────────────────────────────
            st.success("✅ Analysis complete!")

            # Overview card
            health = result.get("health_status", "Unknown")
            severity = result.get("severity", "Low")
            confidence = result.get("confidence", "—")
            plant_name = result.get("plant_name", "Unknown Plant")

            st.markdown(f"""
            <div class="result-card">
                <h3>🌱 Plant Overview</h3>
                <p><strong style="color:#a8e6b0">Plant:</strong> {plant_name}</p>
                <p><strong style="color:#a8e6b0">Status:</strong> {health} &nbsp; {severity_badge(severity)}</p>
                <p><strong style="color:#a8e6b0">Confidence:</strong> {confidence}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Diagnosis card
            issues = result.get("issues_detected", [])
            issues_html = "".join(f"<li>{i}</li>" for i in issues) if issues else "<li>No major issues</li>"
            diagnosis = result.get("diagnosis", "")

            st.markdown(f"""
            <div class="result-card">
                <h3>🔬 Diagnosis</h3>
                <p>{diagnosis}</p>
                <br>
                <strong style="color:#a8e6b0">Issues Detected:</strong>
                <ul>{issues_html}</ul>
            </div>
            """, unsafe_allow_html=True)

            # Causes card
            causes = result.get("causes", [])
            if causes:
                causes_html = "".join(f"<li>{c}</li>" for c in causes)
                st.markdown(f"""
                <div class="result-card">
                    <h3>⚡ Possible Causes</h3>
                    <ul>{causes_html}</ul>
                </div>
                """, unsafe_allow_html=True)

            # Remedies card
            remedies = result.get("remedies", [])
            if remedies:
                remedies_html = "".join(
                    f"<p><strong style='color:#a8e6b0'>💊 {r.get('title','')}</strong><br>{r.get('description','')}</p>"
                    for r in remedies
                )
                urgency = result.get("urgency", "Low")
                st.markdown(f"""
                <div class="result-card">
                    <h3>🌿 Remedies &amp; Treatment</h3>
                    <p><strong style="color:#a8e6b0">Urgency:</strong> {severity_badge(urgency)}</p>
                    <hr class="divider">
                    {remedies_html}
                </div>
                """, unsafe_allow_html=True)

            # Prevention tips
            tips = result.get("prevention_tips", [])
            if tips:
                tips_html = "".join(f"<li>{t}</li>" for t in tips)
                st.markdown(f"""
                <div class="result-card">
                    <h3>🛡️ Prevention Tips</h3>
                    <ul>{tips_html}</ul>
                </div>
                """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        PaudeKaDoctor &nbsp;·&nbsp; Powered by AI &nbsp;·&nbsp; GPT-4o Vision &nbsp;·&nbsp; 
        For informational purposes only
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
