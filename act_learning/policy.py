import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    Conditional Variational Auto-Encoder for ACT.
    Encodes (current_qpos, action_sequence) -> z
    Decodes (z, current_qpos) -> action_sequence (reconstruction)
    During inference, we sample z from prior (N(0,1)).
    """
    def __init__(self, action_dim, latent_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        # Encoder: [Joints, ActionSeq] -> z
        # Input: Action sequence (chunk_size * action_dim) + condition (joints)
        # Simplified: We flatten everything for the MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), # Placeholder input size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(hidden_dim, latent_dim)
        self.z_log_var = nn.Linear(hidden_dim, latent_dim)
        
        # We need a projection specific to input logic. 
        # But ACT usually uses a Transformer Encoder for CVAE as well or just MLP. 
        # Implementation details vary. Here we create a general CMF (Class)
        
    def forward(self, x):
        pass # Implemented inside ACTPolicy for easier integration

class ACTPolicy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.state_dim = args['state_dim'] # qpos dim
        self.action_dim = args['action_dim']
        self.chunk_size = args['chunk_size']
        self.camera_names = args['camera_names']
        self.latent_dim = args['latent_dim']
        self.hidden_dim = args['hidden_dim']
        self.n_heads = args['n_heads']
        self.n_enc_layers = args['n_enc_layers']
        self.n_dec_layers = args['n_dec_layers']
        
        # 1. Backbone (ResNet)
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Identity() # Remove classifier
        self.backbone_dim = 512
        
        # 2. Input Projections
        self.img_proj = nn.Linear(self.backbone_dim, self.hidden_dim)
        self.joint_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.chunk_size, self.hidden_dim))
        self.query_embed = nn.Parameter(torch.zeros(1, self.chunk_size, self.hidden_dim)) # Learned queries
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim)) # For CVAE style encoding
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)

        # 3. Transformer
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            num_encoder_layers=self.n_enc_layers,
            num_decoder_layers=self.n_dec_layers,
            dim_feedforward=self.hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        
        # 4. CVAE Encoder (MLP for simplicity or another Transformer)
        # Using simple MLP for [CLS] token style encoding of action chunk
        self.cvae_encoder = nn.Sequential(
            nn.Linear(self.action_dim * self.chunk_size + self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.z_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.z_log_var = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # 5. Output Head
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1) # Optional: masking

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        qpos: (B, state_dim)
        image: (B, C, H, W) -> Assumes 1 camera for now
        actions: (B, chunk_size, action_dim) -> GT actions for training
        """
        batch_size = qpos.shape[0]
        
        # --- Image Encoding ---
        # image: [B, 3, H, W]
        img_feats = self.backbone(image) # [B, 512]
        img_emb = self.img_proj(img_feats).unsqueeze(1) # [B, 1, H_dim]
        
        # --- Joint Encoding ---
        joint_emb = self.joint_proj(qpos).unsqueeze(1) # [B, 1, H_dim]
        
        # --- CVAE (Training only) ---
        if actions is not None:
            # Flatten actions and concat with qpos
            flat_actions = actions.flatten(start_dim=1)
            cvae_input = torch.cat([flat_actions, qpos], dim=1)
            h = self.cvae_encoder(cvae_input)
            mu = self.z_mean(h)
            log_var = self.z_log_var(h)
            # Reparameterize
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # Inference: Sample from prior
            mu = torch.zeros(batch_size, self.latent_dim).to(qpos.device)
            log_var = torch.zeros(batch_size, self.latent_dim).to(qpos.device)
            z = torch.randn(batch_size, self.latent_dim).to(qpos.device)
        
        # Project z
        z_emb = self.latent_proj(z).unsqueeze(1) # [B, 1, H_dim]
        
        # --- Transformer Input Construction ---
        # Encoder Input: [z, joints, image_features]
        # In typical ACT, they use a DETR backbone which returns feature maps.
        # Here we used ResNet global average pooling for simplicity (Mobile Manipulator constraint).
        # To make it better, we could use spatial features, but let's start simple.
        
        enc_input = torch.cat([z_emb, joint_emb, img_emb], dim=1) # [B, 3, H_dim]
        # Permute for Transformer (Batch First = True)
        
        # Decoder Input: Learned Queries (Chunk Size)
        # Position embeddings are usually added.
        tgt = self.query_embed.repeat(batch_size, 1, 1) # [B, Chunk, H_dim]
        tgt = tgt + self.pos_embed.repeat(batch_size, 1, 1)

        # Transformer Pass
        # memory = encoder_output
        hs = self.transformer(enc_input, tgt) # [B, Chunk, H_dim]
        
        # Prediction
        action_preds = self.action_head(hs) # [B, Chunk, Action_dim]
        
        return action_preds, mu, log_var
