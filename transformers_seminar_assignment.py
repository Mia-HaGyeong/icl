import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def is_paraphrase(sentence1, sentence2):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    model = AutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

    encoded_input = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    similarity_score = torch.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=0)

    threshold = 0.7

    if similarity_score > threshold:
        return "The sentences are paraphrases."
    else:
        return "The sentences are not paraphrases."
        
def main():
    st.title("Paraphrase Detection")

    sentence1 = st.text_input("Enter the first sentence:")
    sentence2 = st.text_input("Enter the second sentence:")

    if sentence1 and sentence2:
        result = is_paraphrase(sentence1, sentence2)
        st.write(result)

if __name__ == '__main__':
    main()
