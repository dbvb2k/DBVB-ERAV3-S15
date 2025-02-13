import torch
from model import DeepSeekForCausalLM, DeepSeekConfig
from transformers import AutoTokenizer
import argparse
import logging
import warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint
    config = checkpoint['config']
    
    # Initialize model with config
    model = DeepSeekForCausalLM(config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to eval mode
    model.eval()
    
    logging.info("Model loaded successfully")
    return model

def generate_text(model, tokenizer, prompt, num_predictions=1, max_length=100, temperature=0.7):
    """Generate text predictions"""
    # Encode prompt with model's max position embeddings
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=model.config.max_position_embeddings,  # Use model's max length
        return_attention_mask=True
    )
    
    # Generate predictions
    predictions = []
    for i in range(num_predictions):
        logging.info(f"Generating prediction {i+1}/{num_predictions}")
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=min(max_length, model.config.max_position_embeddings),  # Respect both limits
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,  # Prevent repetition
                repetition_penalty=1.2    # Penalize repetition
            )
        
        # Decode prediction
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the text
        prediction = prediction.strip()
        prediction = ' '.join(prediction.split())  # Remove multiple spaces
        prediction = prediction.replace(' ,', ',')  # Fix spacing around punctuation
        prediction = prediction.replace(' .', '.')
        
        predictions.append(prediction)
    
    return predictions

def main():
    parser = argparse.ArgumentParser(description='Check DeepSeek model predictions')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_predictions', type=int, default=1, help='Number of predictions to generate')
    parser.add_argument('--prompt', type=str, default=None, help='Input text for prediction')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    
    args = parser.parse_args()
    
    # Set default prompt if none provided
    if args.prompt is None:
        args.prompt = "Once upon a time in a distant galaxy,"
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/cosmo2-tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = load_model(args.checkpoint)
    
    # Generate predictions
    logging.info(f"Generating {args.num_predictions} predictions for prompt: {args.prompt}")
    predictions = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        args.num_predictions,
        args.max_length,
        args.temperature
    )
    
    # Print predictions
    print("\nGenerated Predictions:")
    print("=" * 50)
    for i, pred in enumerate(predictions, 1):
        print(f"\nPrediction {i}:")
        print("-" * 30)
        print(pred)
        print("-" * 30)

if __name__ == "__main__":
    main() 