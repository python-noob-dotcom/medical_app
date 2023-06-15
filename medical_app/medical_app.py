import streamlit as st
from PIL import Image
import torch
from transformers import *

@st.cache(persist=True, allow_output_mutation=True, show_spinner=False, suppress_st_warning=True)
def generate_response(prompt):
    model_id = "Narrativaai/BioGPT-Large-finetuned-chatdoctor"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
    model = AutoModelForCausalLM.from_pretrained(model_id).to('cuda')

    def answer_question(prompt, temperature=0.1, top_p=0.75, top_k=40, num_beams=2, **kwargs):
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=512,
                eos_token_id=tokenizer.eos_token_id
            ).to('cuda')
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return output.split(" Response:")[1]

    return answer_question(prompt=prompt)

def main():
    st.set_page_config(
        page_title="Medical Diagnosis App",
        page_icon="✨",
        layout="centered",
        initial_sidebar_state="auto",
    )

    main_banner = Image.open('D:\Downloads\medical_app\static\main_banner.png')
    top_image = Image.open('D:\Downloads\medical_app\static\doctor_art_1.jpg')
    st.image(main_banner, use_column_width='auto')
    st.sidebar.image(top_image, use_column_width='auto')

    format_type = st.sidebar.selectbox(label='Select a function', options=["Diagnosis", "Treatment Recommendations"])
    user_data , rank= st.text_area('Please enter your name and rank here in the format (name, rank)').split(',')
    button_1 = st.button('Submit!')
    if user_data and rank and button_1:
        if format_type == 'Diagnosis':
            input_text = st.text_area('Please Enter your symptoms here!', height=50)
            button = st.button('Generate your diagnosis')

            if input_text and button:
                with st.spinner('Loading'):
                    response = generate_response("Hi, I will be listing some symptoms I'm facing below, please tell me what condition I may have Symptoms: " + input_text)
                    st.success(response)
            elif button:
                st.warning('Please enter something', icon='⚠')

        elif format_type == 'Treatment Recommendations':
            input_text = st.text_area('Please Enter your symptoms here!', height=50)
            button = st.button('Generate your diagnosis')

            if input_text and button:
                with st.spinner('Loading'):
                    response = generate_response('Hi, I will be listing some symptoms I\'m facing below, please tell me what I should do to fix this! Symptoms:' + input_text)
                    st.success(response)
            elif button:
                st.warning('Please enter something', icon='⚠')

if __name__ == '__main__':
    main()
