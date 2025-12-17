import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CocoCaptionVLMDataset(Dataset):
    """Fixed Dataset for Vision-Language Model training with COCO Captions"""
    
    def __init__(self, dataset_split, transform=None, max_length=38, 
                 tokenizer=None, is_train=True, return_image_id=False):
        """
        Args:
            dataset_split: The dataset split from load_dataset (e.g., coco_dataset["train"])
            transform: Image transformations
            max_length: Maximum text sequence length
            tokenizer: Tokenizer for text processing (REQUIRED)
            is_train: Whether this is training mode
            return_image_id: Whether to return image_id
        """
        if tokenizer is None:
            raise ValueError(
                "Tokenizer is required! Use CLIPTokenizer or AutoTokenizer. "
                "Example: AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')"
            )
        
        self.dataset = dataset_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.return_image_id = return_image_id
        
        # Get special token IDs from tokenizer
        self.eos_token_id = getattr(tokenizer, 'eos_token_id', None)
        self.bos_token_id = getattr(tokenizer, 'bos_token_id', None)
        self.pad_token_id = getattr(tokenizer, 'pad_token_id', None)
        
        # If EOS not defined, try to set it
        if self.eos_token_id is None:
            if hasattr(tokenizer, 'sep_token_id'):
                self.eos_token_id = tokenizer.sep_token_id  # Use [SEP] as EOS for BERT-like
            else:
                self.eos_token_id = 2  # Default for LLaMA and most LMs
        
        if transform is None:
            self.transform = self._get_default_transform(is_train)
        else:
            self.transform = transform
    
    def _get_default_transform(self, is_train):
        """Create appropriate transforms for training vs validation"""
        if is_train:
            return transforms.Compose([
                # TODO  remove the cropping
                transforms.Resize(256),              # Resize shorter edge to 256, maintaining aspect ratio
                transforms.CenterCrop(224),          # Center crop to 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            # Validation: minimal transforms
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:  
            item = self.dataset[idx]
            
            # Load and transform image
            image = item['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image_tensor = self.transform(image)
            
            # Process text
            text = item['text_input']
            text = "Describe the image"+ text
            
            
            # Tokenize text
            text_encoding = self.tokenizer(
                text,
                max_length=self.max_length - 1,  # Reserve 1 space for EOS token
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = text_encoding['input_ids'].squeeze(0)
            attention_mask = text_encoding['attention_mask'].squeeze(0)
            seq_len = (attention_mask == 1).sum().item()
            
            # Add padding to the end of real tokens (afterseq_len = (attention_mask == 1).sum().item()  # Number of real tokens
            
            if seq_len < len(input_ids):  # If there's padding
                # Insert EOS at the end of real tokens (before padding)
                input_ids[seq_len] = self.eos_token_id
                attention_mask[seq_len] = 1  # Mark EOS as real token
            else:
                # Sequence is full, replace last token with EOS
                input_ids[-1] = self.eos_token_id
            #print(self.eos_token_id, "eos_token id ")
            labels = input_ids[1:].clone()  # Shift by 1: [t1, t2, ..., t_n]
            
            # Pad labels to match input_ids length
            # Use -100 as padding in labels (ignored by CrossEntropyLoss)
            labels = torch.cat([labels, torch.tensor([-100], dtype=labels.dtype)])
            labels[labels == 0] = -100

            
            # Mask out padding tokens from original sequence
            # Set label to -100 where original attention_mask is 0 (padding positions)
            labels[attention_mask == 0] = -100
            
            # Also ensure the last token is masked (since it has no target in labels)
            if attention_mask[-1] == 1:  # If last token is not padding
                labels[-1] = -100  # No label for last position in sequence
            
            # Prepare return dictionary
            sample = {
                'image': image_tensor,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,  # ✓ Now properly shifted for next-token prediction
                'text': text
            }
            
            if self.return_image_id:
                sample['image_id'] = item.get('image_id', idx)
            
            return sample
        
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Gracefully skip to next sample instead of crashing
            return self.__getitem__((idx - 1) % len(self))


class CocoCaptionVLMCollator:
    """Simple collator that stacks pre-tokenized data"""
    
    def __init__(self):
        # Collator only stacks tensors, tokenization happens in Dataset
        pass
    
    def __call__(self, batch):
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        
        # Stack pre-tokenized data (no re-tokenization)
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])  # ✓ Include labels
        
        # Get texts
        texts = [item['text'] for item in batch]
        
        # Create batch dictionary
        batch_dict = {
            'images': images,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,  # ✓ Include labels in batch
            'texts': texts
        }
        
        # Add image_id if present
        if 'image_id' in batch[0]:
            batch_dict['image_ids'] = [item['image_id'] for item in batch]
        
        return batch_dict


def create_vlm_dataloaders(coco_dataset, batch_size=32, num_workers=4, 
                           tokenizer=None, max_length=77):
    """Create train dataloader for VLM training"""
    
    if tokenizer is None:
        raise ValueError("Tokenizer is required!")
    
    # Create dataset
    train_dataset = CocoCaptionVLMDataset(
        dataset_split=coco_dataset['train'],
        is_train=True,
        tokenizer=tokenizer,
        max_length=max_length
    )
    # Create VAL dataset (if needed, not used here)
    val_dataset = CocoCaptionVLMDataset(
        dataset_split=coco_dataset['val'],
        is_train=False,
        tokenizer=tokenizer,
        max_length=max_length 
    )
    
    collator = CocoCaptionVLMCollator()
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader
    }
