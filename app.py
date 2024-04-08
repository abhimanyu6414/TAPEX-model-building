import streamlit as st
import pandas as pd
from transformers import TapexTokenizer, BartForConditionalGeneration

# Load the TAPEX tokenizer and model
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

# Function to predict answers
def predict_with_model(model, tokenizer, table, query):
    # Tokenize the query and table
    max_length = model.config.max_position_embeddings
    encoding = tokenizer(table=table, query=query, return_tensors="pt", truncation=True, max_length=max_length)

    # Generate the output
    outputs = model.generate(**encoding)

    # Decode the output
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return prediction

# Streamlit app
def main():
    st.title("TAPEX Table Question Answering")

    # Input query from user
    query = st.text_input("Enter your query:")

    if query:
        # Load your sales dataset
        sales_record = pd.read_csv(r'C:\Users\saima\OneDrive\Desktop\DATA SCIENCE\PROJECT NO-1\10000 Sales Records\10000 Sales Records.csv')

        # Predict the answer
        prediction = predict_with_model(model, tokenizer, sales_record.astype(str), query)

        # Display the prediction
        st.subheader("Prediction:")
        st.write(prediction)

if __name__ == "__main__":
    main()


