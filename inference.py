"""
Vision-Language Model Inference Script
Generates text from images using the trained BasicVLM model
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Optional, List, Tuple, Generator
import argparse
from datetime import datetime


class VLMInference:
    """Handle inference for Vision-Language Model"""
    
    def __init__(self, model_path: str, device: str = 'cuda', 
                 tokenizer=None, max_length: int = 100):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to saved model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            tokenizer: Tokenizer for decoding tokens to text
            max_length: Maximum sequence length
        """
        self.device = device
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Get vocabulary size for output validation
        self.vocab_size = self.model.token_embeds.num_embeddings
        
        # Setup image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_path: str):
        """Load model from checkpoint"""
        from models.vlm import BasicVLM
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Reconstruct model
        config = checkpoint.get('model_config', {})
        model = BasicVLM(
            vocab_size=config.get('vocab_size', 30522),
            embed_dim=config.get('embed_dim', 512)
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"✓ Loaded model from {model_path}")
        return model
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor [1, 3, 224, 224]
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.image_transform(image)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def prepare_initial_tokens(self, prompt: Optional[str] = None) -> torch.Tensor:
        """
        Prepare initial token sequence.
        
        Args:
            prompt: Optional text prompt to start generation
            
        Returns:
            Initial token tensor [1, seq_len]
        """
        if prompt is None:
            # Empty prompt - model will start from image features
            # Use a blank token or start token
            if self.tokenizer and hasattr(self.tokenizer, 'bos_token_id'):
                init_tokens = torch.tensor(
                    [[self.tokenizer.bos_token_id]], 
                    dtype=torch.long, 
                    device=self.device
                )
            else:
                # Fallback: use token 101 (often [CLS] in BERT-based tokenizers)
                init_tokens = torch.tensor([[101]], dtype=torch.long, device=self.device)
        else:
            # Encode prompt
            if self.tokenizer is None:
                raise ValueError("Tokenizer required for prompt encoding")
            
            encoding = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            )
            init_tokens = encoding['input_ids'].to(self.device)
        
        return init_tokens
    
    def generate_token_stream(
        self, 
        images: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None
    ) -> Generator[torch.Tensor, None, None]:
        """
        Generate text tokens one at a time (streaming generation).
        
        Args:
            images: Image tensor [B, 3, H, W]
            input_ids: Initial token IDs [B, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k most probable tokens
            top_p: Keep tokens with cumulative probability >= top_p (nucleus sampling)
            eos_token_id: Token ID that signals end of sequence
            
        Yields:
            Next token tensor [B, 1]
        """
        self.model.eval()
        current_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get model output
                outputs = self.model(images, current_ids, attention_mask=None)
                
                # Get logits for last position
                next_token_logits = outputs[:, -1, :]  # [B, vocab_size]
                
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k, dim=-1
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True, dim=-1
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Find cutoff
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 0] = False  # Keep at least one token
                    
                    # Create mask in original order
                    indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    indices_to_remove.scatter_(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for EOS token
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                
                yield next_token
                
                # Append to sequence
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Prevent sequence from growing too long
                if current_ids.shape[1] >= self.max_length:
                    break
    
    def generate_greedy(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        eos_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate text using greedy decoding (argmax at each step).
        
        Args:
            images: Image tensor [B, 3, H, W]
            input_ids: Initial token IDs [B, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            eos_token_id: Token ID that signals end of sequence
            
        Returns:
            Tuple of (generated_ids, decoded_text_list)
        """
        generated_ids = input_ids.clone()
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(images, generated_ids, attention_mask=None)
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Check for EOS
                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Decode to text
        decoded_texts = self._decode_batch(generated_ids)
        
        return generated_ids, decoded_texts
    
    def generate_beam_search(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        beam_width: int = 3,
        eos_token_id: Optional[int] = None
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate text using beam search for better quality.
        
        Args:
            images: Image tensor [B, 3, H, W]
            input_ids: Initial token IDs [B, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            beam_width: Number of beams to maintain
            eos_token_id: Token ID that signals end of sequence
            
        Returns:
            Tuple of (best_ids, decoded_text_list)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Expand for beam search
        images_expanded = images.repeat_interleave(beam_width, dim=0)
        
        # Initialize beams
        sequences = input_ids.repeat_interleave(beam_width, dim=0)  # [B*beam, seq_len]
        scores = torch.zeros(batch_size, beam_width, device=device)
        
        self.model.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.model(images_expanded, sequences, attention_mask=None)
                next_logits = outputs[:, -1, :]  # [B*beam, vocab_size]
                
                # Reshape for batch processing
                next_logits = next_logits.view(batch_size, beam_width, -1)
                
                # Add previous scores (log probabilities)
                log_probs = F.log_softmax(next_logits, dim=-1)
                scores_expanded = scores.unsqueeze(-1) + log_probs  # [B, beam, vocab]
                
                # Flatten and get top-k
                scores_flat = scores_expanded.view(batch_size, -1)
                top_scores, top_indices = torch.topk(scores_flat, beam_width, dim=-1)
                
                # Update scores
                scores = top_scores
                
                # Get tokens and sequences
                beam_ids = top_indices // self.vocab_size
                token_ids = top_indices % self.vocab_size
                
                # Gather sequences for next iteration
                batch_indices = torch.arange(batch_size, device=device).unsqueeze(-1)
                sequences = sequences[batch_indices * beam_width + beam_ids]
                
                # Append new tokens
                sequences = torch.cat([sequences, token_ids.unsqueeze(-1)], dim=1)
        
        # Return best beam for each sample
        best_ids = sequences.view(batch_size, beam_width, -1)[:, 0, :]
        decoded_texts = self._decode_batch(best_ids)
        
        return best_ids, decoded_texts
    
    def _decode_batch(self, token_ids: torch.Tensor) -> List[str]:
        """
        Decode batch of token sequences to text.
        
        Args:
            token_ids: Token tensor [B, seq_len]
            
        Returns:
            List of decoded strings
        """
        if self.tokenizer is None:
            # Return token IDs as strings if no tokenizer
            return [str(ids.tolist()) for ids in token_ids]
        
        decoded_texts = []
        for ids in token_ids:
            text = self.tokenizer.decode(
                ids.cpu().numpy(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            decoded_texts.append(text)
        
        return decoded_texts
    
    def process_streaming_output(self, token_stream: Generator) -> str:
        """
        Convert token stream to complete text.
        
        Args:
            token_stream: Generator yielding tokens
            
        Returns:
            Complete decoded text
        """
        tokens = []
        for token in token_stream:
            tokens.append(token)
            if self.tokenizer:
                decoded = self.tokenizer.decode(token.cpu().numpy()[0], skip_special_tokens=True)
                print(decoded, end='', flush=True)
        
        print()  # Newline at end
        
        # Combine all tokens
        if tokens:
            combined = torch.cat(tokens, dim=1)
            return self._decode_batch(combined)[0]
        return ""


def main():
    """Main inference function with CLI interface"""
    parser = argparse.ArgumentParser(
        description='VLM Inference - Generate text from images'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--prompt', type=str, default=None,
                       help='Text prompt to start generation')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--method', choices=['greedy', 'beam', 'sampling'], 
                       default='greedy',
                       help='Decoding method')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Temperature for sampling (only for sampling method)')
    parser.add_argument('--top-k', type=int, default=None,
                       help='Top-k filtering (only for sampling method)')
    parser.add_argument('--top-p', type=float, default=None,
                       help='Top-p (nucleus) filtering (only for sampling method)')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda',
                       help='Device to run inference on')
    parser.add_argument('--tokenizer', type=str, default=None,
                       help='Tokenizer to use (e.g., "bert-base-uncased")')
    
    args = parser.parse_args()
    
    # Load tokenizer if specified
    tokenizer = None
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        print(f"✓ Loaded tokenizer: {args.tokenizer}")
    
    # Initialize inference engine
    print("Initializing inference engine...")
    inference = VLMInference(
        model_path=args.model,
        device=args.device,
        tokenizer=tokenizer
    )
    
    # Load image
    print(f"Loading image: {args.image}")
    images = inference.load_image(args.image)
    
    # Prepare initial tokens
    print(f"Preparing initial tokens...")
    input_ids = inference.prepare_initial_tokens(args.prompt)
    
    # Generate
    print(f"\n{'='*60}")
    print(f"Generating text ({args.method} decoding)...")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    if args.method == 'greedy':
        generated_ids, generated_texts = inference.generate_greedy(
            images, input_ids, max_new_tokens=args.max_tokens
        )
        print(f"\nGenerated text:\n{generated_texts[0]}\n")
    
    elif args.method == 'beam':
        generated_ids, generated_texts = inference.generate_beam_search(
            images, input_ids, max_new_tokens=args.max_tokens
        )
        print(f"\nGenerated text:\n{generated_texts[0]}\n")
    
    elif args.method == 'sampling':
        print("Generating (streaming)...\n")
        token_stream = inference.generate_token_stream(
            images, input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        generated_text = inference.process_streaming_output(token_stream)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nGeneration completed in {elapsed:.2f} seconds")


if __name__ == '__main__':
    main()
