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
        
        # Check if Google API key is provided
        if google_api_key:
            # Load the Google Palm model with the provided API key
            llm = GooglePalm(api_key=google_api_key)
            
            # Initialize SmartDataframe with the uploaded data
            smart_df = SmartDataframe(df, config={"llm": llm})
            
            # Create an input text box for user query for SmartDataframe
            user_query_smart = st.text_input("Enter your question for SmartDataframe:")
            
            # Check if the user has entered a query for SmartDataframe
            if user_query_smart:
                # Use the SmartDataframe to get the response
                response_smart = smart_df.chat(user_query_smart)
                
                # Display the response for SmartDataframe
                st.write("Response from SmartDataframe:")
                st.write(response_smart)
        
        # Check if OpenAI API key is provided
        if openai_api_key:
            os.environ['OPENAI_API_KEY'] = openai_api_key
            llm = OpenAI(temperature=0)
            agent = create_csv_agent(
                llm,
                df,
                verbose=False,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            
            # Create an input text box for user query for OpenAI
            user_query_openai = st.text_input("Enter your question for OpenAI:")
            
            # Check if the user has entered a query for OpenAI
            if user_query_openai:
                # Use the OpenAI agent to get the response
                answer_openai = agent.run(user_query_openai)
                
                # Display the response for OpenAI
                st.write("Answer from OpenAI:")
                st.write(answer_openai)

# Run the Streamlit app
if __name__ == "__main__":
    main()

