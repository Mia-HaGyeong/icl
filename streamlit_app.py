
model = SentenceTransformer('sentence-transformers/paraphrase-xlm-r-multilingual-v1')

@st.cache
def get_embedding(sentence):
    return model.encode([sentence])

@st.cache
def compute_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

def main():
    st.title("Paraphrase Detection")

    sentence1 = st.text_input("Enter the first sentence:")
    sentence2 = st.text_input("Enter the second sentence:")

    if sentence1 and sentence2:
        embedding1 = get_embedding(sentence1)
        embedding2 = get_embedding(sentence2)

        similarity_score = compute_similarity(embedding1, embedding2)

        threshold = 0.7

        if similarity_score > threshold:
            st.write("The sentences are paraphrases.")
        else:
            st.write("The sentences are not paraphrases.")

if __name__ == '__main__':
    main()
