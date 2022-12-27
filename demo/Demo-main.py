# Demmo for Emchatbot
# coding: utf-8

# In[1]:

# Main demo interface and methods
import gradio as gr


from transformers import AutoTokenizer, AutoModelForCausalLM

from transformers import AutoModelWithLMHead,AutoTokenizer,pipeline 

tokenizer01 = AutoTokenizer.from_pretrained("Baise/Research_demo_chatbot")

model01 = AutoModelForCausalLM.from_pretrained("Baise/Research_demo_chatbot")


mode_name = 'liam168/trans-opus-mt-zh-en'
model = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)
translation_step1 = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer)

mode_name = 'liam168/trans-opus-mt-en-zh'
model = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)
translation_step2 = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)

#text_zh = input("User:")

def translation_zh_en(text):
    translation_zh_en= translation_step1(text, max_length=400) 
    
    return translation_zh_en
    
#text_en = translation_zh_en(text_zh)
#textx=text_en[0]
#text_en = textx['translation_text']
#print(translation_zh_en(text_zh))



def Emchatbot(text):
    new_user_input_ids = tokenizer01.encode(text+tokenizer01.eos_token, return_tensors='pt')

    bot_input_ids = new_user_input_ids

    # generated a response
    chat_history_ids = model01.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer01.eos_token_id)
    
    return format(tokenizer01.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    #print(format(tokenizer01.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
    

    
#text_en_re = Emchatbot(text_en)
def translation_en_zh(text):
    translation_en_zh = translation_step2(text, max_length=400)
    
    return translation_en_zh
    
#print(translation_en_zh(text_en))


#text_zh = input("User:")

def end_to_end(text):
    text_en = translation_zh_en(text)
    textx=text_en[0]
    text_en = textx['translation_text']
    print(text_en)
    
    text_en_re = Emchatbot(text_en)
    print(text_en_re)

    text_en = translation_en_zh(text_en_re)
    textx=text_en[0]
    text_en = textx['translation_text']
    print(text_en)
    return text_en

#print(end_to_end(text_zh))
#def translation_en_zh(text):
#    translation_en_zh = translation_step2(text, max_length=400)
#    return translation_en_zh

#print(translation_en_zh(text_en))
# 我很伤心，因为我考试不及格。
# 我通过了考试，所以我很高兴。




import gradio as gr


def chat(message, history):
    history = history or []
    response = end_to_end(message)
    history.append((message, response))
    return history, history


iface = gr.Interface(
    chat,
    ["text", "state"],
    ["chatbot", "state"],
    allow_screenshot=False,
    allow_flagging="never",
)
iface.launch(share=True)






