import torch
import torch.nn as nn
import torch.optim as optim
import flare
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5EncoderModel
)
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import os
from rouge_score import rouge_scorer
rscore  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

from transformers import LlamaForCausalLM, AutoTokenizer
from llama_cookbook.configs import train_config as TRAIN_CONFIG
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)

train_config = TRAIN_CONFIG()
train_config.model_name = "meta-llama/Meta-Llama-3.1-8B"
train_config.num_epochs = 1
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.output_dir = "meta-llama-samsum"
train_config.use_peft = True

# Define constants
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
PPO_EPOCHS = 2
SFT_EPOCHS = 1
LEARNING_RATE = 1e-6
CLIP_EPSILON = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            use_cache=False,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            torch_dtype=torch.float16,
        )
        self.value_head = nn.Linear(self.model.config.d_model, 1)
        #self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        outputs = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state.mean(dim=1)
        value = self.value_head(pooled_output)
        #return self.activation(value)
        return value
    
class LoadDataset(Dataset):
    def __init__(self, dataset_name, type_data, tokenizer):
        self.data = load_dataset(dataset_name)[type_data]
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        query, response = item["query"], item["answer"]
        
        inputs = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        targets = self.tokenizer(
            response,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': targets['input_ids'].squeeze(0),
            'query': query,
            'response': response
        }
        
    def retdataframe(self):
        return pd.DataFrame(self.data)
    
class RewardFunction:
    def __init__(self):
        self.reward_model = AutoModelForSequenceClassification.from_pretrained('reward_model_final', num_labels=1)
        self.tokenizer = AutoTokenizer.from_pretrained('reward_model_final')
        self.reward_model.to(DEVICE)
        self.reward_model.eval()
        
    def compute_reward(self, query, response):
        inputs = self.tokenizer(
            text=query,
            text_pair=response,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.reward_model(**inputs)
        
        return outputs.logits.squeeze().item()

class PPOTrainer:
    def __init__(self, actor, critic, tokenizer, learning_rate=LEARNING_RATE, clip_epsilon=CLIP_EPSILON):
        self.actor = actor
        self.critic = critic
        self.tokenizer = tokenizer
        self.optimizer_actor = optim.AdamW(self.actor.parameters(), lr=learning_rate)
        self.optimizer_critic = optim.AdamW(self.critic.parameters(), lr=learning_rate)
        self.reward_fn = RewardFunction()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='none')
        self.clip_epsilon = clip_epsilon
        #self.scheduler_actor = get_linear_schedule_with_warmup(self.optimizer_actor, num_warmup_steps=100, num_training_steps=EPOCHS * SAMPLE_SIZE // BATCH_SIZE)
        #self.scheduler_critic = get_linear_schedule_with_warmup(self.optimizer_critic, num_warmup_steps=100, num_training_steps=EPOCHS * SAMPLE_SIZE // BATCH_SIZE)

    def supervised_fine_tuning(self, dataloader, epochs=SFT_EPOCHS):
        self.actor.train()
        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(dataloader, desc=f"SFT Epoch {epoch+1}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                outputs = self.actor(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                self.optimizer_actor.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.optimizer_actor.step()
                self.scheduler_actor.step()

                progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

    def train(self, dataloader, epochs=EPOCHS, ppo_epochs=PPO_EPOCHS):
        
        #self.supervised_fine_tuning(dataloader, epochs=SFT_EPOCHS)
        self.actor.train()
        self.critic.train()

        for epoch in range(epochs):
            epoch_actor_loss = 0
            epoch_critic_loss = 0
            progress_bar = tqdm(dataloader, desc=f"PPO Epoch {epoch+1}")
            
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                query = batch['query']
                response = batch['response']

                with torch.no_grad():
                    generated_ids = self.actor.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=MAX_LENGTH)

                generated_texts = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
                query = [self.tokenizer.decode(l, skip_special_tokens=True) for l in input_ids]
                
                assert len(generated_texts) == len(query), "Mismatch in lengths of generated texts and queries"
                
                rewards = torch.tensor(
                    [self.reward_fn.compute_reward(que, pred) for que, pred in zip(query, generated_texts)]).to(DEVICE)

                # Compute old log probabilities
                old_outputs = self.actor(input_ids=input_ids, attention_mask=attention_mask, labels=generated_ids)
                
                print("Old outputs:", old_outputs.logits.shape)
                
                old_log_probs = torch.log_softmax(old_outputs.logits, dim=-1).detach()
                old_log_probs = torch.gather(old_log_probs, dim=-1, index=generated_ids.unsqueeze(-1)).squeeze(-1)
                old_log_probs = old_log_probs.sum(dim=-1)

                # Get critic's value estimates
                values = self.critic(input_ids=input_ids, attention_mask=attention_mask).squeeze()
                
                print("Values:", values.shape)
                # Calculate advantages
                advantages = rewards.detach() - values.detach()
                
                print("Advantages:", advantages.shape)
                # PPO Update Loop
                total_actor_loss = 0.0
                total_critic_loss = 0.0

                for _ in range(ppo_epochs):
                    new_outputs = self.actor(
                        input_ids=input_ids, 
                        attention_mask=attention_mask,
                        labels=generated_ids
                    )
                    
                    print("New outputs probs:", new_outputs.logits.shape)
                    new_log_probs = torch.log_softmax(new_outputs.logits, dim=-1)
                    new_log_probs = torch.gather(
                        new_log_probs, 
                        dim=-1, 
                        index=generated_ids.unsqueeze(-1)
                    ).squeeze(-1)
                    new_log_probs = new_log_probs.sum(dim=-1)
                    
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                    
                    new_values = self.critic(input_ids=input_ids, attention_mask=attention_mask).squeeze()
                    critic_loss = nn.MSELoss()(new_values, rewards)
                    #actor_loss = (-torch.min(surr1, surr2)+ 0.5 * critic_loss).mean() 
                    actor_loss = (-torch.min(surr1, surr2)).mean() + 0.5 * critic_loss
                    
                    print("Actor Loss --------------:", actor_loss.item())
                    
                    self.optimizer_actor.zero_grad()
                    
                    print("Start backward actor loss") 
                    actor_loss.backward()
                    print("After backward actor loss") 
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                    self.optimizer_actor.step()

                    #self.scheduler_actor.step()                
                    epoch_actor_loss += actor_loss.item()
                    epoch_critic_loss += critic_loss.item()
                    
                    progress_bar.set_postfix({
                        'actor_loss': epoch_actor_loss / (progress_bar.n + 1),
                        'critic_loss': epoch_critic_loss / (progress_bar.n + 1),
                        'avg_reward': rewards.mean().item()
                    })
                    
        # Save models
        save_path = os.getenv("MODEL_SAVE_PATH", "./saved_models")
        os.makedirs(save_path, exist_ok=True)
        self.actor.save_pretrained(os.path.join(save_path, "actor"))
        print(f"Models saved at {save_path}")
            

def evaluate(actor, critic, dataloader, tokenizer, reward_function, device):
    actor.eval()
    critic.eval()
    total_reward, total_critic_value, count = 0, 0, 0
    results = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=True)
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            generated = actor.generate(input_ids=input_ids, max_length=MAX_LENGTH)
            predicted_texts = [tokenizer.decode(g, skip_special_tokens=True) for g in generated]
            reference_texts = [tokenizer.decode(l, skip_special_tokens=True) for l in labels]

            rewards = [reward_function.compute_reward(ref, pred) for ref, pred in zip(reference_texts, predicted_texts)]
            values = critic(input_ids=input_ids, attention_mask=attention_mask).squeeze()
            
            total_reward += sum(rewards)
            total_critic_value += values.sum().item()
            count += len(rewards)
            
            
            for ref, pred, r, v in zip(reference_texts, predicted_texts, rewards, values.cpu().numpy()):
                scores = rscore.score(ref, pred)
                results.append({
                    "Reference": ref, 
                    "Generated": pred, 
                    "Reward": r,
                    "RougeL": scores["rougeL"].fmeasure,
                    "Predicted Value": float(v)
                })
                
            progress_bar.set_postfix(avg_reward=total_reward / count)

    avg_reward = total_reward / count
    avg_critic_value = total_critic_value / count
    return pd.DataFrame(results), avg_reward, avg_critic_value

def main():
    
    model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            use_cache=False,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            torch_dtype=torch.float16,
        )

    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    actor = model.to(DEVICE)
    critic = Critic().to(DEVICE)
    reward_function = RewardFunction()
    available_datasets = {
    "flare_es_german": "ChanceFocus/flare-german",
    "flare_es_fiqasa": "ChanceFocus/flare-fiqasa",
    "flare_es_instruction_tuning": 'ChanceFocus/flare-es-instruction-tuning',
    "flare_es_stock": "ChanceFocus/flare-es-stock",
    "flare_es_fpb": "ChanceFocus/en-fpb",
    "flare_es_m2sum": "ChanceFocus/m2sum",
    "flare_pubmedsum": "ChanceFocus/pubmedsum",
    "flare_headlines": "ChanceFocus/flare-headlines",
    "flare_finqa": "ChanceFocus/flare-finqa",
    "flare_convfinqa": "ChanceFocus/flare-convfinqa",
    "flare_ner": "ChanceFocus/flare-ner",
    "flare_cikm": "ChanceFocus/flare-sm-cikm",
    "flare_finer_acl": "ChanceFocus/flare-sm-acl",
    }
    for key_name, dataset_name in available_datasets:
        train_dataset = LoadDataset(dataset_name=dataset_name, type_data="train", tokenizer=tokenizer)
        test_dataset = LoadDataset(dataset_name=dataset_name, type_data="test", tokenizer=tokenizer)
        validation_dataset = LoadDataset(dataset_name=dataset_name, type_data="valid", tokenizer=tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        val_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)

    
        # Train with SFT and then PPO
        ppo_trainer = PPOTrainer(actor, critic, tokenizer)
        actor, critic = ppo_trainer.train(train_dataloader)

        print("Evaluating AFTER fine-tuning...")
        df_after_test, final_reward_test, final_critic_value_test = evaluate(
            actor, critic, test_dataloader, tokenizer, reward_function, DEVICE
        )
        df_after_test.to_csv(f"test_results_ {key_name}.csv", index=False)
        df_after_val, final_reward_val, final_critic_value_val = evaluate(
            actor, critic, val_dataloader, tokenizer, reward_function, DEVICE
        )
        df_after_val.to_csv(f"val_results_ {key_name}.csv", index=False)
        # Print improvement summary
        print("\n--- Training Results Summary ---")
if __name__ == "__main__":
    main() 
