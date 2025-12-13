import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models.vlm import BasicVLM


class BasicVLMTrainer:
    def __init__(self, model, train_loader, val_loader=None, 
                 device='cuda', learning_rate=1e-4, max_epochs=10,
                 log_wandb=False, wandb_project="basic-vlm-training"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_epochs = max_epochs
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=max_epochs * len(train_loader)
        )
        
        # Loss function - ignore padding tokens
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Setup logging
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(project=wandb_project)
            wandb.watch(self.model)
        
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get model outputs
            outputs = self.model(images, input_ids)
            
            # Prepare targets for loss calculation
            # We predict the next token, so shift targets by 1
            # Also need to account for image tokens
            num_img_tokens =1
            B,  D = self.model.image_projector(
                self.model.vision_encoder(images)
            ).size()
            
            # Create target labels (shift input_ids by 1)
            # Image tokens don't have language targets, so we use -100 (ignore_index)
            image_targets = torch.full((B, num_img_tokens), -100, device=self.device)
            text_targets = input_ids
            
            # Concatenate targets
            targets = torch.cat([image_targets, text_targets], dim=1)
            
            # Calculate loss
            # outputs shape: [B, seq_len, vocab_size]
            # targets shape: [B, seq_len]
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            
            # Calculate token-level loss for logging
            batch_tokens = (targets != -100).sum().item()
            batch_loss = loss.item() * batch_tokens
            
            total_loss += batch_loss
            total_tokens += batch_tokens
            
            # Update progress bar
            avg_loss = total_loss / max(total_tokens, 1)
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
            })
            
            # Log to wandb
            if self.log_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0]
                })
        
        return total_loss / max(total_tokens, 1)
    
    @torch.no_grad()
    def validate(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        
        pbar = tqdm(self.val_loader, desc="Validation")
        for batch in pbar:
            images = batch['images'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            
            # Forward pass
            outputs = self.model(images, input_ids)
            
            # Prepare targets
            B, num_img_tokens, D = self.model.image_projector(
                self.model.vision_encoder(images)
            ).size()
            
            image_targets = torch.full((B, num_img_tokens), -100, device=self.device)
            text_targets = input_ids
            targets = torch.cat([image_targets, text_targets], dim=1)
            
            # Calculate loss
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )
            
            batch_tokens = (targets != -100).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
            
            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / max(total_tokens, 1)
        return avg_loss
    
    def train(self):
        print(f"Starting training for {self.max_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Log epoch results
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f'best_model_epoch_{epoch+1}.pt')
                    print(f"Saved best model with val loss: {val_loss:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            # Log to wandb
            if self.log_wandb:
                log_dict = {'epoch': epoch+1, 'train_loss': train_loss}
                if val_loss is not None:
                    log_dict['val_loss'] = val_loss
                wandb.log(log_dict)
        
        print("Training completed!")
        if self.log_wandb:
            wandb.finish()
    
    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.max_epochs,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'vocab_size': self.model.token_embeds.num_embeddings,
                'embed_dim': self.model.token_embeds.embedding_dim
            }
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Checkpoint loaded from {path}")


if __name__ == "__main__":
    
    # Use the same tokenizer as in your data loader
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tokenizer.vocab_size
    
    model = BasicVLM(
        vocab_size=vocab_size,
        embed_dim=512
    )
    
    # Get dataloaders (from your existing code)
    from lavis.datasets.builders import load_dataset
    import os
    
    os.environ['cache_root'] = "/home/ha/.cache/lavis/coco"
    
    coco_dataset = load_dataset("coco_caption")
    
    from dataloader import create_vlm_dataloaders
    
    print("\nCreating dataloaders...")
    dataloaders = create_vlm_dataloaders(
        coco_dataset,
        batch_size=16,
        num_workers=2,
        tokenizer=tokenizer,
        max_length=50
    )
    
    train_loader = dataloaders['train']
    val_loader = dataloaders.get('val', None)
    
    # Create trainer
    trainer = BasicVLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate=1e-4,
        max_epochs=10,
        log_wandb=False  # Set to True if you want wandb logging
    )
    
    # Start training
    trainer.train()