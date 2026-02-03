import torch
import torch.nn as nn
import torch.nn.functional as F

# https://tintn.github.io/Implementing-Vision-Transformer-from-Scratch/


class PatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]

        # calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2

        # projection layer
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)

        # create learnable cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))

        # create learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size = x.shape[0]

        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  

        # prepend cls token to the input
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x
    

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, head_size, dropout, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, head_size, bias=bias)
        self.key = nn.Linear(hidden_size, head_size, bias=bias)
        self.value = nn.Linear(hidden_size, head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, seq_length, hidden_size)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / (self.hidden_size ** 0.5)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # compute attention output
        out = torch.matmul(weights, value)
        return (out, weights)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]

        assert self.hidden_size % self.num_attention_heads == 0, "Hidden size must be divisible by number of attention heads"
        self.head_size = self.hidden_size // self.num_attention_heads
        self.qkv_bias = config["qkv_bias"]

        self.heads = nn.ModuleList([])
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.hidden_size,
                self.head_size, 
                config["dropout_rate"], 
                self.qkv_bias
            )
            self.heads.append(head)

        self.out_projection = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x, output_attentions=False):
        att_outputs = [head(x) for head in self.heads]
        att_out = torch.cat([out for out, _ in att_outputs], dim=-1)
        att_out = self.out_projection(att_out)
        att_out = self.dropout(att_out)
        if not output_attentions:
            return (att_out, None)
        else:
            attention_probs = torch.stack([attention_probs for _, attention_probs in att_outputs], dim=1)
            return (att_out, attention_probs)
        

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["mlp_dim"])
        self.fc2 = nn.Linear(config["mlp_dim"], config["hidden_size"])
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
    

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layernorm1 = nn.LayerNorm(config["hidden_size"])
        self.mlp = MLP(config)
        self.layernorm2 = nn.LayerNorm(config["hidden_size"])

    def forward(self, x, output_attentions=False):
        att_output, attention_probs = self.attention(self.layernorm1(x), output_attentions=output_attentions)

        # skip connection
        x = x + att_output

        mlp_output = self.mlp(self.layernorm2(x))

        # skip connection
        x = x + mlp_output

        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
        

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = TransformerBlock(config)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
        

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.hidden_size = config["hidden_size"]
        self.num_classes = config["num_classes"]

        self.embedding = Embeddings(config)
        self.encoder = Encoder(config)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits, take the [CLS] token's output as features for classification
        logits = self.classifier(encoder_output[:, 0, :])
        # Return the logits and the attention probabilities (optional)
        if not output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)