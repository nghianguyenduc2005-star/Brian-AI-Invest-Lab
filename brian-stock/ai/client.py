import streamlit as st

try:
    from google import genai
except Exception:
    genai = None

def get_key():
    try:
        return str(st.secrets.get("GEMINI_API_KEY","")).strip()
    except Exception:
        return ""

def ask(prompt, model=None, image_path=None):
    key=get_key()
    if not key:
        return None, "Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets."
    if genai is None:
        return None, "Thiếu package google-genai."
    try:
        client=genai.Client(api_key=key)
        response=client.models.generate_content(
            model=model or st.session_state.get("ai_model","gemini-2.5-flash"),
            contents=prompt,
        )
        return response.text, None
    except Exception as e:
        return None, f"AI lỗi: {e}"
