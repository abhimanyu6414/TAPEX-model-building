from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd
import joblib
import pickle
sales_record = pd.read_csv(r"C:\Users\saima\OneDrive\Desktop\DATA SCIENCE\PROJECT NO-1\10000 Sales Records\10000 Sales Records.csv")
sales_record.head()

sales_record = sales_record.astype(str)

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

# Specify your query
query = "What was the total revenue in the North region for 2023?"

# Truncate the input to fit within the model's maximum sequence length
max_length = model.config.max_position_embeddings
encoding = tokenizer(table=sales_record, query=query, return_tensors="pt", truncation=True, max_length=max_length)

# Generate the output
outputs = model.generate(**encoding)

# Decode the output
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))


#model with loaded tokenizer
tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

sales_record = sales_record.astype(str)


def predict_with_model(model, tokenizer, table, query):
    # Tokenize the query and table
    max_length = model.config.max_position_embeddings
    encoding = tokenizer(table=table, query=query, return_tensors="pt", truncation=True, max_length=max_length)

    # Generate the output
    outputs = model.generate(**encoding)

    # Decode the output
    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return prediction

while True:
    query = input("Enter a query (or type 'exit' to stop): ")
    if query.lower() == 'exit':
        break

    # Predict using the model and table
    prediction = predict_with_model(model, tokenizer,sales_record , query)

    # Print the prediction
    print(f"Query: {query}")
    print(f"Prediction: {prediction}")
    print()






