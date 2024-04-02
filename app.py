import streamlit as st
import pandas as pd
import io
from pandasai import SmartDataframe
from pandasai.llm import GooglePalm
from pandasai.openai import OpenAI, AgentType, create_csv_agent
import os
import time

# Define the Streamlit app
def main():
    st.title("PALM and OPEN AI CSV/EXCEL CHATBOT")
    
    # Create input field for Google API key
    google_api_key = st.text_input("Enter your Google API key:")
    
    # Create input field for OpenAI API key
    openai_api_key = st.text_input("Enter your OpenAI API key", type="password")
    
    # Check if Google API key is provided
    if google_api_key:
        # Load the Google Palm model with the provided API key
        llm = GooglePalm(api_key=google_api_key)
        
        # Define function to load SmartDataframe
        def load_smart_dataframe(data_file):
            # Load the SmartDataframe
            df = SmartDataframe(data_file, config={"llm": llm})
            return df
        
        # Create file uploader component for SmartDataframe
        uploaded_file_smart = st.file_uploader("Upload file for SmartDataframe", type=['csv', 'xlsx'])
        
        # Check if file is uploaded for SmartDataframe
        if uploaded_file_smart is not None:
            # Check file type for SmartDataframe
            if uploaded_file_smart.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': # XLSX file
                # Read XLSX file as DataFrame
                xls = pd.ExcelFile(uploaded_file_smart)
                df = pd.read_excel(xls)
                # Convert DataFrame to CSV format
                csv_file = io.StringIO()
                df.to_csv(csv_file, index=False)
                # Load CSV data into a DataFrame
                csv_file.seek(0)
                df = pd.read_csv(csv_file)
            else: # CSV file
                # Load data into a DataFrame
                df = pd.read_csv(uploaded_file_smart)
            
            # Initialize SmartDataframe with the uploaded data
            smart_df = load_smart_dataframe(df)
            
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
        agent = None
        
        # Define function to create OpenAI agent
        def create_openai_agent():
            agent = create_csv_agent(
                llm,
                uploaded_file_openai,
                verbose=False,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            return agent
        
        # Create file uploader component for OpenAI
        uploaded_file_openai = st.file_uploader("Upload CSV file for OpenAI", type=["csv", "xlsx"])
        
        # Check if file is uploaded for OpenAI
        if uploaded_file_openai is not None:
            agent = create_openai_agent()
            
            # Predefined questions for OpenAI
            predefined_questions = ["How many rows are there in the dataset?", "Explain the dataset."]
            selected_question_openai = st.selectbox("Select a question for OpenAI", ["Select a question for OpenAI"] + predefined_questions) 
            custom_question_openai = st.text_input("Or ask a custom question for OpenAI")
            
            if st.button("Ask OpenAI"):
                if selected_question_openai != "Select a question for OpenAI":
                    query_openai = selected_question_openai
                elif custom_question_openai.strip() != "":
                    query_openai = custom_question_openai.strip()
                else:
                    st.warning("Please select a predefined question or ask a custom question for OpenAI.")
                    return

                start_openai = time.time()
                answer_openai = agent.run(query_openai)
                end_openai = time.time()
                st.write("Answer from OpenAI:")
                st.write(answer_openai)
                st.write(f"Answer from OpenAI (took {round(end_openai - start_openai, 2)} seconds)")

# Run the Streamlit app
if __name__ == "__main__":
    main()
