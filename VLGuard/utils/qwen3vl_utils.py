import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

def model_inference(engine, model, image, prompt, processor, max_new_tokens):
    
    # image_tensor =  processor.preprocess([image], return_tensors='pt')['pixel_values'].to(torch.float16).cuda()
    
    # if model.config.mm_use_im_start_end:
    #     inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
    # else:
    #     inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    
    # conv_mode = 'llava_v1'
    # conv = conv_templates[conv_mode].copy()
    # conv.append_message(conv.roles[0], inp)
    # conv.append_message(conv.roles[1], None)
    # prompt = conv.get_prompt()
    # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, 
                                       max_new_tokens=max_new_tokens, 
                                       do_sample=False,
                                       temperature=1,
                                       min_new_tokens=1,)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
    predicted_answers = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # print(predicted_answers)
    # predicted_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return predicted_answers

# def load_model(model_path, args=None):
#     tokenizer, model, image_processor, context_len = load_llava_model(model_path=model_path, model_base=None, model_name='llava', 
#                                                                       attn_implementation='flash_attention_2', torch_dtype='float16', device_map='cuda',)
#     processor = image_processor
#     return model, tokenizer, processor


def load_model(model_path):
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype="float16", device_map="cuda", 
        torch_dtype=torch.bfloat16, 
        # attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor