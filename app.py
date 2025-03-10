import json
import random
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer




intents = json.load(open('intents.json'))
tags=[]
prompts=[]

for intent in intents['intents']:
    for p in intent['prompt']:
        prompts.append(p)
        tags.append(intent['tag'])
    

vector = TfidfVectorizer()
prompt_scaled = vector.fit_transform(prompts)   


#building model

Bot = LogisticRegression(max_iter=100000)
Bot.fit(prompt_scaled,tags)


#testing the model

def ChatBot(input_message):
     input_message = vector.transform([input_message])
     pred_tag = Bot.predict(input_message)[0]

     for intent in intents['intents']:
        if intent['tag'] == pred_tag:
             response = random.choice(intent['response'])
             return response
 

st.markdown(
    """
    <h1 style='text-align: center; color: blue;'>Welcome to Australia</h1>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
st.markdown(
    "<p style='color: purple; font-size: 18px;'>I am your assistant Kristy, How may I help you?</p>",
    unsafe_allow_html=True,
)


if p := st.chat_input("Enter your message here"):
    # Display user message in chat message container
    st.chat_message("user").markdown(p)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": p})

    response = f" Kristy :"+ ChatBot(p)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})