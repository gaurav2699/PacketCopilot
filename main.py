import streamlit as st
import os
import time

st.set_page_config(page_title='Packet Copilot', page_icon='🗣️')

DEFAULT_SYSTEM_MESSAGE = """
        You are a copilot for packet analysis from the trace file. Help the user.
"""

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'system_message' not in st.session_state:
    st.session_state['system_message'] = DEFAULT_SYSTEM_MESSAGE

if 'streaming_enabled' not in st.session_state:
    st.session_state['streaming_enabled'] = False



st.markdown('#### Upload the document')
packetFile = st.file_uploader(label='Upload either a PCAP or ETL file to chat', accept_multiple_files=False, type=['pcap','etl'])
st.markdown('#### Chat')
if packetFile == None:
    st.markdown('#### Waiting for the trace file to be uploaded...')
else:
    with st.spinner('Getting the Trace...'):
        with open(f'{packetFile.name}', 'wb') as f:
            f.write(packetFile.read())
        os.remove(f'{packetFile.name}')
    with st.chat_message(name='assistant'):
        st.markdown('Chat with me..')
    for message in st.session_state.messages:
        with st.chat_message(name=message['role']):
            st.markdown(message['content'])
    if prompt := st.chat_input('Enter your prompt'):
        st.session_state.messages.append({'role' : 'user', 'content' : prompt})
        with st.chat_message(name='user'):
            st.markdown(prompt)
