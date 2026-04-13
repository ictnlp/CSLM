import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Union
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from datasets import load_dataset
from transformers import LlamaForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_utils import get_last_checkpoint

import os
import logging
import sys
import re

sys.path.append("..")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default="", 
        metadata={"help": "Path to the training data."})
    prompt_template_name: str = field(
        default="alpaca",
        metadata={"help": "prompt_template_name"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the tokenized data"},
    )
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    val_set_size: int = field(
        default=2000,
        metadata={"help": "val_set_size"},
    )
    preprocessing_num_workers: int = field(
        default=100,
        metadata={"help": "preprocessing_num_workers for tokenizing"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "num_epochs"},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "learning_rate"},
    )
    output_dir: str = field(
        default="",
        metadata={"help": "output_dir"},
    )
    train_on_inputs: bool = field(
        default=False,
        metadata={"help": "if False, masks out inputs in loss"},
    )
    initial_global_step: int = field(
        default=0,
        metadata={"help": "initial_global_step"}
    )
    save_total_limit: int = field(
        default=3,
        metadata={"help": "save_total_limit"}
    )


def remove_space_in_unit_seq(text):
    # remove space between ">" and "<"
    text = re.sub(r'>\s+<', '><', text)
    return text


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa



def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    logger.info(f"Set save_total_limit={training_args.save_total_limit}")
    logger.info(f"Set load_best_model_at_end={training_args.load_best_model_at_end}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
    )

    
    # Llama3: AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # tokenizer.pad_token_id = (
    #     0  # unk. we want this to be different from the eos token
    # )
    
    # llama3 pad: 128002
    tokenizer.pad_token_id = 128002

    #Extend vocab for [question], [answer]
    if "[question]" not in tokenizer.get_vocab():
        logger.info(f"Add special unit tokens [question], [answer] to tokenizer.vocab")
        new_tokens = ["[question]", "[answer]"]
        tokenizer.add_tokens(new_tokens)

    #resize embedding
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < tokenizer.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result


    def generate_and_tokenize_prompt(data_point):
        if type(data_point["prompt"]) == list:
            # 多轮对话
            human_text_list = data_point["prompt"]
            llm_text_list = data_point["response"]
            messages = []
            for human_text, llm_text in zip(human_text_list, llm_text_list):
                # apply template
                messages.append({"role": "user", "content": human_text})
                messages.append({"role": "assistant", "content": llm_text})
        else:
            human_text = data_point["prompt"]
            llm_text = data_point["response"]
            # apply template
            messages = [{"role": "user", "content": human_text}, {"role": "assistant", "content": llm_text}]
        messages = tokenizer.apply_chat_template(messages, tokenize=False)
        # 去除结尾最后一个的"<|start_header_id|>assistant<|end_header_id|>"
        if messages.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"):
            messages = messages[:-len("<|start_header_id|>assistant<|end_header_id|>\n\n")]

        # tokenize
        tokenized_prompt = tokenize(messages)
        # if not train_on_inputs, set labels before "assistant" to -100
        if not training_args.train_on_inputs:
            seg_list = ["<|start_header_id|>", "assistant", "<|end_header_id|>"]
            seg_list = [tokenizer.convert_tokens_to_ids(x) for x in seg_list]
            user_seg_list = ["<|start_header_id|>", "user", "<|end_header_id|>"]
            user_seg_list = [tokenizer.convert_tokens_to_ids(x) for x in user_seg_list]
            input_begin = 0
            input_end = 0
            for i, token_id in enumerate(tokenized_prompt["input_ids"]):
                # 连续len(seg_list)个token_id为seg_list，将其与其之前的label设为-100
                if token_id == seg_list[0] and tokenized_prompt["input_ids"][i:i+len(seg_list)] == seg_list:
                    if i + len(seg_list) >= len(tokenized_prompt["input_ids"]):
                        print(f"seg_list exceeds input_ids length, i={i}")
                        break
                    input_end = i
                    tokenized_prompt["labels"][input_begin:input_end] = [-100] * (input_end - input_begin)
                if token_id == user_seg_list[0] and tokenized_prompt["input_ids"][i:i+len(user_seg_list)] == user_seg_list:
                    input_begin = i
            # 开头若为<|begin_of_text|>，将其label设为-100
            if tokenized_prompt["input_ids"][0] == tokenizer.convert_tokens_to_ids("<|begin_of_text|>"):
                tokenized_prompt["labels"][0] = -100

        return tokenized_prompt


    if data_args.data_path.endswith(".json") or data_args.data_path.endswith(".jsonl"):
        data = load_dataset("json", data_files=data_args.data_path)
    else:
        data = load_dataset(data_args.data_path)

    tokenized_cache_file_names = {
        "train":os.path.join(training_args.cache_dir, 'tokenized', 'train', 'processed_train.arrow'),
        "test":os.path.join(training_args.cache_dir, 'tokenized', 'valid', 'processed_valid.arrow'),
    }

    if training_args.val_set_size > 0:
        train_val = data["train"].train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=42
        )
        train_val_data = (
            train_val.map(
                generate_and_tokenize_prompt,
                batched=False,
                num_proc=training_args.preprocessing_num_workers,
                load_from_cache_file=True,
                cache_file_names=tokenized_cache_file_names,
                desc=f"generate_and_tokenize_prompt",
                )
        )
        train_data = train_val_data["train"]
        val_data = train_val_data["test"]

    else:
        train_data = data["train"].map(
                generate_and_tokenize_prompt,
                batched=False,
                num_proc=training_args.preprocessing_num_workers,
                load_from_cache_file=True,
                cache_file_names=tokenized_cache_file_names,
                desc=f"generate_and_tokenize_prompt",
                )
        val_data = None


    data_collator = DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )

    # torch_empty_cache_steps=100
    training_args.torch_empty_cache_steps = 100

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_data if training_args.do_train else None, 
        eval_dataset=val_data if training_args.do_eval else None, 
        data_collator=data_collator,
    )

    if training_args.initial_global_step != 0:
        logger.info(f"Set initial global step={training_args.initial_global_step}")
        trainer.state.global_step = training_args.initial_global_step

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_data)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_data))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(val_data)
        metrics["eval_samples"] = min(max_eval_samples, len(val_data))
        try:
            import math
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    train()