import streamlit as st
import pandas as pd
import io
from pandasai import SmartDataframe
from pandasai.llm import GooglePalm
from langchain.llms.openai import OpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import os

# Function to create SmartDataframe and get response
def smart_dataframe_function(data_file, user_query):
    llm = GooglePalm(api_key="your_google_api_key_here")
    df = pd.read_csv(data_file)
    smart_df = SmartDataframe(df, config={"llm": llm})
    response_smart = smart_df.chat(user_query)
    return response_smart

# Function to create OpenAI agent and get response
def openai_function(data_file, user_query):
    os.environ['OPENAI_API_KEY'] = "your_openai_api_key_here"
    llm = OpenAI(temperature=0)
    agent = create_csv_agent(
        llm,
        data_file,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    answer_openai = agent.run(user_query)
    return answer_openai

# Define the Streamlit app
def main():
    st.title("PALM and OPEN AI CSV/EXCEL CHATBOT")

    # Input fields for API keys
    google_api_key = st.text_input("Enter your Google API key:")
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload dataset file", type=['csv', 'xlsx'])

    # Check if dataset file is uploaded and contains data
    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)

        # Check if the DataFrame is not empty
        if not df.empty:
            # Ask a question to be answered by both OpenAI and SmartDataframe
            user_query = st.text_input("Ask a question:")

            # Check if user has asked a question
            if user_query:
                # Display the response from SmartDataframe
                if google_api_key:
                    response_smart = smart_dataframe_function(io.StringIO(uploaded_file.getvalue().decode("utf-8")), user_query)
                    st.write("Response from SmartDataframe:")
                    st.write(response_smart)
                else:
                    st.write("Please provide your Google API key to use SmartDataframe.")

                # Display the response from OpenAI
                if openai_api_key:
                    response_openai = openai_function(io.StringIO(uploaded_file.getvalue().decode("utf-8")), user_query)
                    st.write("Answer from OpenAI:")
                    st.write(response_openai)
                else:
                    st.write("Please provide your OpenAI API key to use OpenAI.")
        else:
            st.write("The uploaded file is empty. Please upload a file with data.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
