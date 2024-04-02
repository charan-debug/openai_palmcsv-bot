import streamlit as st
import pandas as pd
import io
from pandasai import SmartDataframe
from pandasai.llm import GooglePalm
from langchain.llms.openai import OpenAI
from langchain_experimental.agents import create_csv_agent
from langchain.agents.agent_types import AgentType
import os
import time

# Define the SmartDataframe function
def smart_dataframe_function(df, user_query):
    # Load the Google Palm model
    llm = GooglePalm(api_key="your_google_api_key_here")
    # Initialize SmartDataframe with the provided data
    smart_df = SmartDataframe(df, config={"llm": llm})
    # Use SmartDataframe to get the response
    response_smart = smart_df.chat(user_query)
    return response_smart

# Define the OpenAI function
def openai_function(df, user_query):
    # Set OpenAI API key
    os.environ['OPENAI_API_KEY'] = "your_openai_api_key_here"
    # Load OpenAI model
    llm = OpenAI(temperature=0)
    # Create OpenAI agent
    agent = create_csv_agent(
        llm,
        df,
        verbose=False,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    # Use OpenAI agent to get the response
    answer_openai = agent.run(user_query)
    return answer_openai

# Define the Streamlit app
def main():
    st.title("PALM and OPEN AI CSV/EXCEL CHATBOT")

    # Create input field for Google API key
    google_api_key = st.text_input("Enter your Google API key:")

    # Create input field for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

    # Create file uploader component for dataset
    uploaded_file = st.file_uploader("Upload dataset file", type=['csv', 'xlsx'])

    # Check if dataset file is uploaded
    if uploaded_file is not None:
        # Load the dataset into a DataFrame
        if uploaded_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': # XLSX file
            # Read XLSX file as DataFrame
            xls = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(xls)
        else: # CSV file
            # Load data into a DataFrame
            df = pd.read_csv(uploaded_file)

        # Ask a question to be answered by both OpenAI and SmartDataframe
        user_query = st.text_input("Ask a question:")

        # Check if user has asked a question
        if user_query:
            # Display the response from SmartDataframe
            st.write("Response from SmartDataframe:")
            st.write(smart_dataframe_function(df, user_query))
            
            # Display the response from OpenAI
            st.write("Answer from OpenAI:")
            st.write(openai_function(df, user_query))

# Run the Streamlit app
if __name__ == "__main__":
    main()
