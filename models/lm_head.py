class LMHead(nn.Module):
    """LM head with layer normalization for better training stability"""
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, final_states):
        normalized = self.norm(final_states)
        logits = self.lm_head(normalized)
        return logits