from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from evaluate import load
import torch
import logging
import warnings
from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
import os

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(filename="testingLLMs_boolq.log", encoding="utf-8", level=logging.DEBUG)

class boolq:
    def __init__(self, model, tokenizer) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.tokenizer.pad_token = self.tokenizer.eos_token

        accelerator = Accelerator()


        self.model, self.tokenizer = accelerator.prepare(self.model, self.tokenizer)
        # Load dataset
        self.dataset = load_dataset("super_glue", 'boolq', trust_remote_code=True)


        self.tokenized_dataset = self.dataset.map(
            self.__preprocess_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )

    def __preprocess_function(self, examples):
        """Preprocess the data for T5"""

        inputs = [
            f"answer yes or no: {question} passage: {passage}"
            for question, passage in zip(examples["question"], examples["passage"])
        ]


        targets = [
            "yes" if label else "no"
            for label in examples["label"]
        ]

  
        model_inputs = self.tokenizer(
            inputs,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=8,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

        model_inputs["labels"] = labels["input_ids"]


        model_inputs["labels"] = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        return model_inputs

    def train(self):
        """Training with LoRA applied to the model"""

        # Enable LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, 
            r=16,  # LoRA rank
            lora_alpha=32,  # Scaling factor
            lora_dropout=0.1,  # Dropout applied to LoRA layers
            target_modules=["q", "v"]  # Apply LoRA to attention query and value projection layers
        )

        self.model = get_peft_model(self.model, peft_config)

        self.model.gradient_checkpointing_enable()

       
        for param in self.model.parameters():
            param.requires_grad = True  #

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )

      
        accuracy_metric = load('accuracy')

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            
            logits = logits[0] if isinstance(logits, tuple) else logits
            logits=torch.tensor(logits)
            labels = labels[0] if isinstance(labels, tuple) else labels
            labels = torch.tensor(labels)
            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions.view(-1)
            labels = labels.view(-1)
 
            mask = labels != -100
            predictions = predictions[mask]
            labels = labels[mask]
            
            return accuracy_metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir='./flan_t5_boolq_lora',
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=250,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            logging_dir='./boolq_training_logs',
            logging_steps=10,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",  # Use accuracy as the metric for best model selection
            fp16=False,
            optim="adamw_torch",
            gradient_checkpointing=True,
            eval_steps=250,
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics  
        )

        return trainer.train()

    def __predict(self):
        predictions = []
        for example in tqdm(self.dataset['validation'], desc="Predicting"):
            input_text = f"answer yes or no: {example['question']} passage: {example['passage']}"
            inputs = self.tokenizer(
                input_text,
                max_length=256,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(**inputs)

            pred = self.tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
            predictions.append(1 if pred == "yes" else 0)

        return predictions

    def compute_metric(self, metric='accuracy'):
        metric = load(metric)
        predictions = self.__predict()
        return metric.compute(
            predictions=predictions,
            references=self.dataset['validation']['label']
        )

if __name__ == "__main__":
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    model_name = "google/flan-t5-small"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Train with LoRA
    boolq_model = boolq(model=model, tokenizer=tokenizer)
    boolq_model.train()

    # Evaluate
    result = boolq_model.compute_metric()
    print(f"Model accuracy: {result}")
a