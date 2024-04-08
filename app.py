import streamlit as st
from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")
sales_record = pd.read_csv(r"C:\Users\saima\OneDrive\Desktop\DATA SCIENCE\PROJECT NO-1\10000 Sales Records\10000 Sales Records.csv")
sales_record = sales_record.astype(str)


def predict_with_model(model, tokenizer, table, query):

    max_length = model.config.max_position_embeddings
    encoding = tokenizer(table=table, query=query, return_tensors="pt", truncation=True, max_length=max_length)


    outputs = model.generate(**encoding)


    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return prediction


def main():
    st.title("TAPEX Table Question Answering")


    query = st.text_input("Enter your query:")

    if query:

        prediction = predict_with_model(model, tokenizer, sales_record, query)


        st.subheader("Prediction:")
        st.write(prediction)

if __name__ == "__main__":
    main()

