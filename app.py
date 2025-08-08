import streamlit as st
from openai import OpenAI

# Hugging Face OpenAI-compatible client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=st.secrets["HF_TOKEN"],  # Replace with your HF token
)

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.markdown("Ask me anything!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**Bot:** {msg['content']}")

# Chat input box at the bottom
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get response from model
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b:novita",  # Model from Hugging Face
        messages=st.session_state.messages
    )

    # Extract bot reply
    reply = completion.choices[0].message.content

    # Add bot reply to history
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # Rerun to show the updated conversation
    st.rerun()
