import streamlit as st
from layers import *
import torch
from torch.autograd import Variable
# from transformers import AutoTokenizer, RobertaForMaskedLM
    
# Load models
noise = Variable(torch.FloatTensor(64, 12))
noise = noise.to('cpu')
generator = Stage1Generator(noise)
generator.load_state_dict(torch.load('netG.pth'))
caug = CAug(device='cpu')

# preprocess text
def preprocess_text(input):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    model = RobertaForMaskedLM.from_pretrained(
        'roberta-base', output_hidden_states=True).to('cpu')
    model.eval()
    try:
        encoded_dict = tokenizer.encode_plus(
            input,
            add_special_tokens=True,
            max_length=128,  # This is changed.
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        input_ids = encoded_dict["input_ids"]
        attention_masks = encoded_dict["attention_mask"]
        # FOR ROBERTA EMBEDDING
        with torch.no_grad():
            # token embeddings:
            outputs = model(encoded_dict['input_ids'].to(
                'cpu'), encoded_dict['attention_mask'].to('cpu'))
            hidden_states = outputs[1]
            token_embeddings = torch.stack(hidden_states, dim=0)

            # sentence embeddings:
            token_vecs = hidden_states[-2][0]
            sentence_embedding = torch.mean(token_vecs, dim=0)
            sentence_embedding = sentence_embedding.view(1, -1)
            print(f"Input sentence: {input}")
            print(f"RoBERTa embedding: {sentence_embedding}")
            return sentence_embedding
    except Exception as e:
        # print(f"Error in {bird_type}/{file}")
        print(e)    

def generate_samples(caug_val):
    with torch.no_grad():
        sample = generator(caug_val)
        
    return sample

st.title('Implementing RoBERTa text-embedding technique in StackGAN model')

with st.form('input_form'):
    input_text = st.text_input('Text description')
    submit_button = st.form_submit_button('GENERATE')
    
if submit_button and input_text is not None:
    preprocessed_text = preprocess_text(input_text)
    c_0_hat, mu, logvar = caug(preprocessed_text)
    sample = generate_samples(c_0_hat)
    
    print(sample)