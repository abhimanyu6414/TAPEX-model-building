import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, BartForConditionalGeneration


def predict(table_path, query):
  """
  Predicts answer to a question using the TAPEX model on a given table.

  Args:
      table_path: Path to the CSV file containing the table data.
      query: The question to be answered.

  Returns:
      The predicted answer as a string.
  """
  sales_record = pd.read_csv(table_path)
  sales_record = sales_record.astype(str)  # Ensure string type for tokenizer

  max_length = model.config.max_position_embeddings
  encoding = tokenizer(table=sales_record, query=query, return_tensors="pt", truncation=True, max_length=max_length)

  outputs = model.generate(**encoding)

  prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
  return prediction


# **Model and Tokenizer Loading (Deployment-Friendly):**

try:
  tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
  model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")
except Exception as e:
  st.error(f"An error occurred loading the TAPEX model and tokenizer: {e}")
  # Consider adding logging or error handling mechanisms here


st.title("TAPEX Table Q&A App")

uploaded_file = st.file_uploader("Upload Sales Data (CSV)", type="csv")

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.write(df)

  query = st.text_input("Ask a question about the sales data:")

  if query:
    prediction = predict(uploaded_file.name, query)
    st.write(f"**Your Question:** {query}")
    st.write(f"**Predicted Answer:** {prediction}")
else:
  st.info("Please upload a CSV file containing your sales data.")

