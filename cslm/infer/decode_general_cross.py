import sys
import torch
from typing import List
import argparse
import logging
import json
from tqdm import tqdm
import os
import re
from transformers import LlamaForCausalLM, AutoTokenizer, GenerationConfig


logging.basicConfig()
logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prompt_en = "Please answer the questions in the user's speech in English."
prompt_zh = "用中文回答用户语音中的问题。"

TEMPLATE_EN = "{prompt} This is input: <sosp>{src_units}<eosp>."
TEMPLATE_ZH = "{prompt} 这是输入: <sosp>{src_units}<eosp>。"

device = torch.device('cuda')


def read_unit(unit_file):
    with open(unit_file, 'r') as f:
        unit_seqs = f.readlines()
    unit_seqs = [unit_seq.strip() for unit_seq in unit_seqs]
    return unit_seqs


def extract_text_between_tags(text, tag1='LLM] :', tag2='<eoa>'):
    pattern = f'{re.escape(tag1)}(.*?){re.escape(tag2)}'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        response = match.group(1)
    else:
        response = ""
    return response


class LLMInference:
    def __init__(
        self,
        lang: str,
        model_name_or_path: str,
        output_dir: str=None,
        max_len_input=4096,
        max_new_tokens=4096,
        ):

        self.lang = lang
        self.template = TEMPLATE_EN if lang == "en" else TEMPLATE_ZH
        self.max_len_input = max_len_input
        self.max_new_tokens = max_new_tokens

        #model
        self.model = LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
            )

        self.model.half()

        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        #tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # self.tokenizer.pad_token_id = 128002
        # self.tokenizer.padding_side = "left"
        # self.tokenizer.model_max_length = self.max_len_input

        self.output_dir = output_dir


    def preprocess(
        self,
        unit_seq: str,
        tokenizer,
    ):
        # apply template
        prompt = prompt_en if self.lang == "en" else prompt_zh
        instruction = self.template.format(prompt=prompt, src_units=unit_seq)
        response_prefix = ""
        messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": response_prefix}]
        messages = tokenizer.apply_chat_template(messages, tokenize=False)
        if messages.endswith("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"):
            messages = messages[:-len("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")]
        return messages


    def postprocess(
        self,
        response: str,
    ):
        # return {"tq": tq, "ta": ta, "ua": ua}
        if self.lang == "zh":
            result = extract_text_between_tags(response + "<eoa>", tag1="这是输入: <sosp>", tag2="<eoa>")
        else:
            result = extract_text_between_tags(response + "<eoa>", tag1="This is input: <sosp>", tag2="<eoa>")
        return {"result": result}


    def forward(
        self, 
        prompts: List[str]
    ):
        with torch.no_grad():
            #preprocess
            preprocessed_prompts = []
            for prompt in prompts:
                preprocessed_prompts.append(self.preprocess(prompt, self.tokenizer))

            input_ids = self.tokenizer(preprocessed_prompts, return_tensors="pt", padding=True).input_ids
            for input_id in input_ids:
                if input_id[-1] == 2:
                    input_id = input_id[:, :-1]

            input_ids = input_ids.to(device)
            if input_ids.shape[1] > self.max_len_input:
                logger.warning(f"Input length {input_ids.shape[1]} exceeds the maximum length {self.max_len_input}. Truncated to {self.max_len_input}.")
                input_ids = input_ids[:, :self.max_len_input]

            #generate conifg:
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id).to(device)
            config_dict = {
                "temperature": 1.0,
                "top_p": 0.7,
                "do_sample": True,
                "max_new_tokens": self.max_new_tokens,
            }
            generation_config = GenerationConfig(
                **config_dict,
                attention_mask=attention_mask,
            )

            generated_ids = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            generated_ids = generated_ids.sequences
            responses = self.tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)

            #postprocess
            responses = [self.postprocess(x) for x in responses]

            #save repsonses
            with open(f"{self.output_dir}/responses.json", 'a') as f:
                for r in responses:
                    print("Response:\n", r["result"])
                    json_line = json.dumps(r, ensure_ascii=False)
                    f.write(json_line+'\n')

            # print(f"Response json is saved in {self.output_dir}/responses.json")

        return 16000

    def __call__(self, input):
        return self.forward(input)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--unit", type=str, required=True)
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-len-input", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer = LLMInference(
        lang=args.lang,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
    )

    logger.info("Reading unit sequences...")
    unit_seqs = read_unit(args.unit)

    logger.info("Inferring...")
    for units in unit_seqs:
        # 去除所有空格
        units = units.replace(" ", "")
        infer([units])
