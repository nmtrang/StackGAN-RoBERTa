import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.autograd import Variable
from transformers import AutoTokenizer, RobertaForMaskedLM
from layers import Stage1Generator, Stage2Generator
# import engine


def load_model():
    generator1 = Stage1Generator()
    generator2 = Stage2Generator(generator1)
    # discriminator = Stage1Discriminator()
    # Define the path to the pre-trained model file
    netG_path_stage1 = "../output/model/stage1/netG.pth"
    netG_path_stage2 = "../output/model/stage2/netG.pth"
    # netD_path = "../output/model/stage1_modified/netD.pth"
    
    # Load the pre-trained model
    
    # netG_1 = torch.load(netG_path_stage1, map_location=torch.device('cpu'))
    netG_2 = torch.load(netG_path_stage2, map_location=torch.device('cpu'))
    # st.write(netG_1['model_state_dict'])
    # st.write(netG_2['model_state_dict'])
    generator2.load_state_dict(netG_2['model_state_dict'])
    
    # generator1.eval()
    generator2.eval()
    # discriminator.eval()
    
    return generator2

def embed_text(text):
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    model = RobertaForMaskedLM.from_pretrained(
        'roberta-base', output_hidden_states=True).to('cpu')
    model.eval()
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,  # This is changed.
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
        truncation=True,
    )
    with torch.no_grad():
        # token embeddings:
        outputs = model(encoded_text['input_ids'].to('cpu'), encoded_text['attention_mask'].to('cpu'))
        hidden_states = outputs[1]
        token_embeddings = torch.stack(hidden_states, dim=0)

        # sentence embeddings:
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        sentence_embedding = sentence_embedding.view(1, -1).to('cpu')
        
        # st.write(sentence_embedding)
        
        return sentence_embedding

def output_img(fake_img):
    im = fake_img.cpu().detach().numpy()
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    # print("im", im.shape)
    im = np.transpose(im, (1, 2, 0))
    # print("im", im.shape)
    im = Image.fromarray(im)
    st.image(im)


def main():
    netG_2 = load_model()
    # load_model()

    # ------------- STREAMLIT APP HERE -------------
    with st.form(key="my_form"):
        text_input = st.text_input(label="Text description")
        submit_button = st.form_submit_button(label="Generate image!")

    if submit_button:
        # st.spinner(text="In progress...")
        sentence_embedding = embed_text(text_input)
        noise = torch.autograd.Variable(torch.FloatTensor(1, 100))
        noise.data.normal_(0, 1)
        # _, fake_image, mu, logvar = netG_1(sentence_embedding, noise)
        gen1_fake_img, gen2_fake_img, mu, logvar = netG_2(sentence_embedding, noise)
        gen1_fake_img = gen1_fake_img.squeeze(0)
        gen2_fake_img = gen2_fake_img.squeeze(0)
        st.markdown('##### Stage 1 result')
        output_img(gen1_fake_img)
        st.markdown('##### Stage 2 result')
        output_img(gen2_fake_img)


if __name__ == "__main__":
    main()
