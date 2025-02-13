import os
import torch
from torch.utils.data import Dataset, DataLoader
from model import DeepSeekConfig, DeepSeekForCausalLM, MoEConfig
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from typing import Optional, List
import logging
from tqdm.auto import tqdm
import time
from datetime import timedelta
from datasets import load_dataset, concatenate_datasets, interleave_datasets
import glob
import re
from torch.cuda.amp import autocast, GradScaler
from torch.nn.functional import scaled_dot_product_attention
import warnings
from itertools import chain
from torch.utils.data import IterableDataset

# Constants 
BATCH_SIZE = 8  # Increased from 4
MAX_SEQUENCE_LENGTH = 512  # Reduced from 1024
CHECKPOINT_SAVE_INTERVAL = 500  # Less frequent saving
SAMPLE_GENERATION_INTERVAL = 500  # Less frequent sampling

# Constants
TOTAL_STEPS = 40000
PROGESS_STEPS = 50
DATA_ELEMENTS = 2000
NUM_DASHES = 100
MIN_TOKENS = 10
MAX_TOKENS = 2048
VOCAB_SIZE = 50304
DEVICE = "cpu"  # Force CPU usage

# Update constants
LEARNING_RATE = 3e-4  # Reduced from 6e-4
WARMUP_STEPS = 4000   # Increased from 2000
GRAD_ACCUM_STEPS = 4  # Increased from 1
GRAD_CLIP = 1.0      # Increased from 0.5

# Remove unnecessary validations and checks
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

class StreamingDataset(IterableDataset):
    def __init__(self, tokenizer, block_size=512):
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # Add high-quality datasets
        datasets = [
            load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", streaming=True)["train"],
            load_dataset("databricks/databricks-dolly-15k", streaming=True)["train"],
            load_dataset("Abirate/english_quotes", streaming=True)["train"],
            load_dataset("b-mc2/sql-create-context", streaming=True)["train"],
            load_dataset("squad", streaming=True)["train"],
            load_dataset("tatsu-lab/alpaca", streaming=True)["train"]  # Instruction dataset
        ]
        
        # Adjust weights for better balance
        self.dataset = interleave_datasets(
            datasets,
            probabilities=[0.35, 0.2, 0.1, 0.1, 0.1, 0.15],  # Adjusted weights for 6 datasets
            stopping_strategy="first_exhausted"
        )
        
        # More aggressive quality filtering
        def quality_filter(example):
            text = (
                example.get('text') or 
                example.get('content') or 
                example.get('instruction') or 
                example.get('question') or
                example.get('context') or
                ''
            )
            # Stricter quality checks
            return (
                len(text) > 200 and           # Longer texts
                text.count('.') >= 2 and      # Multiple sentences
                len(set(text.split())) > 50   # More vocabulary diversity
            )
        
        self.dataset = self.dataset.filter(quality_filter)
        
        # Enable shuffling with buffer
        self.dataset = self.dataset.shuffle(seed=46, buffer_size=10000)
    
    def __iter__(self):
        iterator = iter(self.dataset)
        buffer = []
        
        while True:
            try:
                # Get next example
                example = next(iterator)
                
                # Handle different text field names
                text = (
                    example.get('text') or 
                    example.get('content') or 
                    example.get('instruction') or 
                    example.get('code') or 
                    ''
                )
                
                # Tokenize text
                tokens = self.tokenizer.encode(text)
                
                # Add to buffer
                buffer.extend(tokens)
                
                # Process buffer into chunks
                while len(buffer) >= self.block_size:
                    chunk = buffer[:self.block_size]
                    buffer = buffer[self.block_size:]
                    
                    # Create input and target sequences
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    
                    yield x, y
                    
            except StopIteration:
                # Reset iterator when exhausted
                iterator = iter(self.dataset)
                buffer = []

def collate_batch(batch):
    """Custom collate function to pad sequences in a batch."""
    # Find max length in the batch
    max_len = max(x[0].size(0) for x in batch)
    
    # Initialize tensors for inputs and targets
    batch_size = len(batch)
    inputs = torch.full((batch_size, max_len), fill_value=0, dtype=torch.long)  # 0 is assumed to be pad_token_id
    targets = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)  # -100 is ignored by CrossEntropyLoss
    
    # Fill in the tensors with actual values
    for i, (input_seq, target_seq) in enumerate(batch):
        seq_len = input_seq.size(0)
        inputs[i, :seq_len] = input_seq
        targets[i, :seq_len] = target_seq
    
    return inputs, targets

def save_checkpoint(model, optimizer, scheduler, step, loss, path):
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }, path)

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    logging.info(f"Loading checkpoint from {path}, Please wait...")
    checkpoint = torch.load(path, weights_only=False)  # Explicitly set weights_only
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['step'], checkpoint['loss']

def get_latest_checkpoint() -> tuple[Optional[str], Optional[int]]:
    """Returns tuple of (checkpoint_path, step_number) or (None, None) if no checkpoints exist"""
    checkpoints = glob.glob("checkpoints/*.pt")
    if not checkpoints:
        return None, None
    
    # Print all checkpoints for debugging
    print(f"\nTotal checkpoints found: {len(checkpoints)}")
    print("Extracting steps from checkpoints...")
    
    steps = []
    for ckpt in checkpoints:
        # Extract step number from filename
        match = re.search(r'model_step_(\d+)_loss_', ckpt)
        if match:
            step_num = int(match.group(1))
            steps.append((step_num, ckpt))
            print(f"+ Extracted step {step_num} from {ckpt}")
    
    if steps:
        latest_step, latest_ckpt = max(steps, key=lambda x: x[0])
        print(f"\nSelected checkpoint:")
        print(f"  Path: {latest_ckpt}")
        print(f"  Step: {latest_step}")
        return latest_ckpt, latest_step
    return None, None

def generate_sample(model, tokenizer, device, prompt="Once upon a time"):
    model.eval()
    try:
        with torch.no_grad():
            # Tokenize with proper padding
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings,
                return_attention_mask=True,
                add_special_tokens=True
            )
            
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
            
            # Decode
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        model.train()
        return generated_text
        
    except Exception as e:
        logging.error(f"Error during text generation: {str(e)}")
        model.train()
        return f"Error generating text: {str(e)}"

class DataLoaderLite:
    """Simple data loader"""
    def __init__(self, B=4, T=512):
        self.B = B
        self.T = T
        
    def next_batch(self):
        x = torch.randint(0, VOCAB_SIZE, (self.B, self.T))
        y = torch.randint(0, VOCAB_SIZE, (self.B, self.T))
        return x, y

def display_model_summary(model):
    """Display model parameters summary including total, trainable params and model size"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info("Model Parameters Summary:")
    logging.info(f"{'='*NUM_DASHES}")
    logging.info(f"Total Parameters:     {total_params:,}")
    logging.info(f"Trainable Parameters: {trainable_params:,}")
    logging.info(f"Model Size:           {total_params * 4 / (1024**2):.2f} MB")
    logging.info(f"{'='*NUM_DASHES}")

def generate_final_samples(model, tokenizer, device, num_samples=5, prompts=None):
    """Generate and display multiple samples after training completion"""
    
    if prompts is None:
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a distant galaxy,",
            "The most important scientific discovery of the 21st century was",
            "In the year 2100, humanity finally achieved",
            "The secret to happiness is"
        ]
    
    print("\n" + "="*NUM_DASHES)
    print("FINAL MODEL PREDICTIONS")
    print("="*NUM_DASHES)
    
    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(prompts[:num_samples], 1):
            try:
                print(f"\nSample {i} (Prompt: '{prompt}')")
                print("-" * NUM_DASHES)
                
                # Generate multiple samples and pick the best one
                num_candidates = 3
                best_text = None
                max_quality_score = -1
                
                for _ in range(num_candidates):
                    # Tokenize
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=model.config.max_position_embeddings,
                        return_attention_mask=True
                    ).to(device)
                    
                    # Generate with stricter parameters
                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=150,  # Shorter length
                        min_length=50,   # Ensure minimum length
                        temperature=0.8,
                        top_p=0.92,
                        top_k=50,
                        do_sample=True,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.pad_token_id,
                        bos_token_id=tokenizer.bos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        no_repeat_ngram_size=3,  # Prevent repetition
                        repetition_penalty=1.2   # Penalize repetition
                    )
                    
                    # Decode and clean up text
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean up the text
                    generated_text = generated_text.strip()
                    generated_text = ' '.join(generated_text.split())  # Remove multiple spaces
                    generated_text = generated_text.replace(' ,', ',')  # Fix spacing around punctuation
                    generated_text = generated_text.replace(' .', '.')
                    
                    # Simple quality checks
                    quality_score = 0
                    if len(generated_text.split()) > 20:  # Reasonable length
                        quality_score += 1
                    if '.' in generated_text:  # Has proper sentences
                        quality_score += 1
                    if not any(c*3 in generated_text for c in '.,'):  # No repetitive punctuation
                        quality_score += 1
                    if len(set(generated_text.split())) > 10:  # Vocabulary diversity
                        quality_score += 1
                    
                    if quality_score > max_quality_score:
                        max_quality_score = quality_score
                        best_text = generated_text
                
                # Print the best generation
                if best_text and max_quality_score >= 2:
                    print(best_text)
                else:
                    print("[Generation filtered due to low quality]")
                print("-" * NUM_DASHES)
                
            except Exception as e:
                print(f"Error generating sample {i}: {str(e)}")
    
    model.train()
    print("\n" + "="*NUM_DASHES + "\n")

def train_model(total_steps: int = TOTAL_STEPS, additional_steps: Optional[int] = None):
    try:
        # Starting the training process
        logging.info("="*NUM_DASHES)
        logging.info("Starting the training process, Please wait...")
        logging.info("="*NUM_DASHES)

        # Setup device
        logging.info("Setting up device, Please wait...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")
        torch.backends.cudnn.benchmark = True
        
        # Create checkpoints directory
        logging.info("Creating checkpoints directory, Please wait...")
        os.makedirs("checkpoints", exist_ok=True)
        logging.info("Checkpoints directory created successfully")
        logging.info(" ")
        
        # Initialize model with DeepSeek config
        logging.info("Initializing DeepSeek model, Please wait...")
        moe_config = MoEConfig(
            num_experts=8,
            num_experts_per_tok=2,
            expert_size=2048,  # Reduced from 4096 for smaller model
            capacity_factor=1.25
        )

        # Working config generated by Cursor        
        # config = DeepSeekConfig(
        #     vocab_size=VOCAB_SIZE,
        #     hidden_size=256,
        #     intermediate_size=512,
        #     num_hidden_layers=6,
        #     num_attention_heads=8,
        #     num_key_value_heads=4,
        #     hidden_act="silu",
        #     max_position_embeddings=512,
        #     initializer_range=0.02,
        #     rms_norm_eps=1e-5,
        #     moe_config=moe_config,
        #     num_latent_heads=4,
        #     latent_size=32
        # )
        
        # Config used by Rohan Shravan
        config = DeepSeekConfig(
            vocab_size=VOCAB_SIZE,
            hidden_size=1024,        # Increased from 768
            intermediate_size=2048,  # Increased from 1536
            num_hidden_layers=32,    # Increased from 30
            num_attention_heads=16,  # Increased from 8
            num_key_value_heads=8,   # Increased from 4
            hidden_act="silu",
            max_position_embeddings=512,
            initializer_range=0.02,
            rms_norm_eps=1e-5,
            moe_config=moe_config,
            num_latent_heads=8,      # Increased from 4
            latent_size=64          # Increased from 32
        )
                
        model = DeepSeekForCausalLM(config).to(device)
        
        logging.info("DeepSeek Model initialized successfully")
        logging.info(" ")

        # Display model parameters
        display_model_summary(model)
        
        # Initialize tokenizer with padding token
        logging.info("Initializing tokenizer with padding token, Please wait...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Also ensure other special tokens are set
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            tokenizer.bos_token_id = tokenizer.eos_token_id
            
        # Update model config with tokenizer's special tokens
        config.pad_token_id = tokenizer.pad_token_id
        config.bos_token_id = tokenizer.bos_token_id
        config.eos_token_id = tokenizer.eos_token_id
        
        logging.info("Tokenizer initialized successfully")
        logging.info(" ")
        
        # Initialize dataset and dataloader
        logging.info("Initializing dataset and dataloader, Please wait...")
        train_dataset = StreamingDataset(tokenizer, block_size=MAX_SEQUENCE_LENGTH)
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=0,  # Streaming dataset works better with single worker
            pin_memory=True,
            collate_fn=collate_batch
        )
        logging.info("Dataset and dataloader initialized successfully")
        logging.info(" ")
        
        train_iter = iter(train_loader)
        
        # Initialize optimizer and scheduler
        logging.info("Initializing optimizer and scheduler, Please wait...")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True
        )
        
        logging.info("Optimizer and scheduler initialized successfully")
        logging.info(" ")

        # Determine start step and total steps
        start_step = 0
        if additional_steps is not None:
            # Try to load the latest checkpoint
            latest_ckpt, latest_step = get_latest_checkpoint()
            if latest_ckpt is not None:
                print(f"\nResuming from checkpoint: {latest_ckpt}")
                print(f"Last completed step: {latest_step}")
                checkpoint = torch.load(latest_ckpt, weights_only=False)  # Explicitly set weights_only
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_step = latest_step + 1
                total_steps = start_step + additional_steps
                print(f"Will train from step {start_step} to {total_steps}\n")
            else:
                print("No checkpoint found. Starting from step 0.")
                total_steps = additional_steps
        
        # Initialize scheduler with correct number of steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        # Load scheduler state if resuming
        if additional_steps is not None and latest_ckpt is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Training loop
        logging.info("Starting training loop, Please wait...")
        logging.info("="*NUM_DASHES)
        model.train()
        for step in range(start_step, total_steps):
            t0 = time.time()
            
            # Update data loading in training loop
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
                
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss / GRAD_ACCUM_STEPS  # Scale loss for accumulation
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            # Log progress
            if step % PROGESS_STEPS == 0:
                dt = time.time() - t0
                tokens_per_sec = (BATCH_SIZE * MAX_SEQUENCE_LENGTH) / dt
                lr = scheduler.get_last_lr()[0]
                print(f'Step {step} | Loss: {loss.item():.4f} | lr: {lr:.2e} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_sec:.2f}')
            

            # Generate sample text every 500 steps
            if step > 0 and (step % SAMPLE_GENERATION_INTERVAL == 0 or step == total_steps - 1):
                print(f"\n{'='*NUM_DASHES}")
                print(f"Generating text sample at step {step}")
                print(f"{'-'*NUM_DASHES}")
                sample_text = generate_sample(model, tokenizer, device)
                print(f"Generated text:\n{sample_text}")
                print(f"{'-'*NUM_DASHES}\n")
                model.train()  # Ensure model is back in training mode
            
            # Save checkpoint every 500 steps and at the final step
            if step > 0 and (step % CHECKPOINT_SAVE_INTERVAL == 0 or step == total_steps - 1):
                # Format loss for filename (e.g., 2.345 becomes "2_34")
                loss_str = f"{loss.item():.2f}".replace('.', '_')
                checkpoint_path = f"checkpoints/model_step_{step}_loss_{loss_str}.pt"
                print(f"Saving checkpoint to {checkpoint_path}")
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                    'config': config,
                }, checkpoint_path)
                print(f"Checkpoint saved successfully\n")
            
            # Add warmup restart every 10000 steps
            if step > 0 and step % 10000 == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LEARNING_RATE
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=1000,
                    num_training_steps=10000
                )
            
        # After training loop completes, generate final samples
        logging.info("Generating final samples...")
        generate_final_samples(model, tokenizer, device)
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_steps', type=int, default=TOTAL_STEPS,
                      help='Total number of steps for fresh training')
    parser.add_argument('--additional_steps', type=int,
                      help='Number of additional steps when resuming training')
    args = parser.parse_args()
    
    if args.additional_steps is not None:
        train_model(additional_steps=args.additional_steps)
    else:
        train_model(total_steps=args.total_steps) 

    logging.info("Training completed successfully !!!")