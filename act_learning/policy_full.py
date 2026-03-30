import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import math

class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal positional encodings for 2D image features.
    Standard implementation from DETR.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x = tensor_list 
        mask = torch.zeros((x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class ACTPolicyBaseline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.state_dim = args['state_dim']
        self.action_dim = args['action_dim']
        self.chunk_size = args['chunk_size']
        self.latent_dim = args['latent_dim']
        self.hidden_dim = args['hidden_dim']
        self.n_heads = args['n_heads']
        
        # --- 1. Backbone (ResNet18) ---
        full_resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(full_resnet.children())[:-2]) 
        self.backbone_dim = 512
        
        # Positional Embedding (2D) for Image Features
        self.pos_embed = PositionEmbeddingSine(num_pos_feats=self.hidden_dim // 2, normalize=True)
        
        # --- 2. CVAE Encoder (Improved: Transformer-based) ---
        # [수정됨] 기존 MLP 대신 Transformer를 사용하여 시퀀스 정보를 처리합니다.
        # Input tokens: [CLS] + [Joint] + [Action_1] + ... + [Action_Chunk]
        self.cvae_cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.cvae_action_proj = nn.Linear(self.action_dim, self.hidden_dim)
        self.cvae_joint_proj = nn.Linear(self.state_dim, self.hidden_dim)
        
        # CVAE용 Positional Embedding (Learnable 1D PE)
        # Length = 1(CLS) + 1(Joint) + Chunk_Size
        self.cvae_pos_embed = nn.Parameter(torch.zeros(1, 1 + 1 + self.chunk_size, self.hidden_dim))
        
        cvae_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=4, dim_feedforward=self.hidden_dim*2, 
            dropout=0.1, batch_first=True
        )
        self.cvae_transformer = nn.TransformerEncoder(cvae_layer, num_layers=2) # 2-layer is usually enough for CVAE
        
        self.z_mean = nn.Linear(self.hidden_dim, self.latent_dim)
        self.z_log_var = nn.Linear(self.hidden_dim, self.latent_dim)
        
        # --- 3. Main Transformer Components ---
        self.input_proj = nn.Conv2d(self.backbone_dim, self.hidden_dim, kernel_size=1)
        self.joint_proj = nn.Linear(self.state_dim, self.hidden_dim)
        self.latent_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        
        # Query Embeds (Learnable input for Decoder)
        self.query_embed = nn.Parameter(torch.zeros(1, self.chunk_size, self.hidden_dim))
        self.tgt_pos_embed = nn.Parameter(torch.zeros(1, self.chunk_size, self.hidden_dim))

        # Main Transformer
        self.transformer = nn.Transformer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            num_encoder_layers=args['n_enc_layers'],
            num_decoder_layers=args['n_dec_layers'],
            dim_feedforward=self.hidden_dim*4,
            dropout=0.1,
            batch_first=True
        )
        
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, qpos, image, actions=None, is_pad=None):
        """
        qpos: (B, state_dim)
        image: (B, 3, H, W)
        actions: (B, chunk_size, action_dim) - Training only
        """
        batch_size = qpos.shape[0]
        device = qpos.device
        
        # -------------------------------
        # 1. CVAE Encoding (Training Only)
        # -------------------------------
        if actions is not None:
            # Prepare Tokens
            # [B, 1, H]
            cls_token = self.cvae_cls_token.expand(batch_size, -1, -1)
            # [B, 1, H]
            joint_token = self.cvae_joint_proj(qpos).unsqueeze(1)
            # [B, Chunk, H]
            action_tokens = self.cvae_action_proj(actions) 
            
            # Concat: [CLS, Joint, Action_1, ..., Action_N]
            cvae_input = torch.cat((cls_token, joint_token, action_tokens), dim=1)
            
            # Add Positional Embedding
            # cvae_input length matches self.cvae_pos_embed
            cvae_input = cvae_input + self.cvae_pos_embed[:, :cvae_input.shape[1], :]
            
            # Transformer Forward
            cvae_out = self.cvae_transformer(cvae_input)
            
            # Take CLS token output for latent distribution
            cls_out = cvae_out[:, 0, :]
            
            mu = self.z_mean(cls_out)
            log_var = self.z_log_var(cls_out)
            std = torch.exp(0.5 * log_var)
            z = mu + torch.randn_like(std) * std
        else:
            mu = torch.zeros(batch_size, self.latent_dim).to(device)
            log_var = torch.zeros(batch_size, self.latent_dim).to(device)
            z = torch.randn(batch_size, self.latent_dim).to(device)

        # -------------------------------
        # 2. Main Policy Inputs
        # -------------------------------
        # (a) Image Features
        features = self.backbone(image)
        pos = self.pos_embed(features)
        
        src = self.input_proj(features)
        src_flat = src.flatten(2).permute(0, 2, 1) # [B, SeqLen, H]
        pos_flat = pos.flatten(2).permute(0, 2, 1)
        
        # (b) Joint Features
        joint_emb = self.joint_proj(qpos).unsqueeze(1)
        # Joint position embedding is typically 0 or learned. Keep 0 for baseline simplicity.
        joint_pos = torch.zeros_like(joint_emb)
        
        # (c) Latent Features
        z_emb = self.latent_proj(z).unsqueeze(1)
        z_pos = torch.zeros_like(z_emb)

        # -------------------------------
        # 3. Transformer Pass
        # -------------------------------
        # Encoder Input: [z, joint, image_tokens...]
        enc_input = torch.cat([z_emb, joint_emb, src_flat], dim=1)
        enc_pos = torch.cat([z_pos, joint_pos, pos_flat], dim=1)
        
        # Decoder Input: Learnable Query
        tgt = self.query_embed.repeat(batch_size, 1, 1)
        tgt_pos = self.tgt_pos_embed.repeat(batch_size, 1, 1)
        
        # Note: In vanilla nn.Transformer, we add pos embedding to input directly
        hs = self.transformer(enc_input + enc_pos, tgt + tgt_pos)
        
        # -------------------------------
        # 4. Output
        # -------------------------------
        action_preds = self.action_head(hs) # [B, Chunk, Action_Dim]
        
        return action_preds, mu, log_var

# --- 사용 예시 (Dummy Data) ---
if __name__ == "__main__":
    # Hyperparameters
    args = {
        'state_dim': 7,          # e.g., 7 DoF Arm
        'action_dim': 7,         # e.g., 7 DoF Arm Velocity/Position
        'chunk_size': 20,        # Action Chunk Size
        'latent_dim': 32,
        'hidden_dim': 256,
        'n_heads': 4,
        'n_enc_layers': 4,
        'n_dec_layers': 4
    }
    
    model = ACTPolicyBaseline(args)
    model.cuda() # GPU가 있다면
    
    # Dummy Inputs
    B = 2
    dummy_qpos = torch.randn(B, args['state_dim']).cuda()
    dummy_image = torch.randn(B, 3, 224, 224).cuda()
    dummy_actions = torch.randn(B, args['chunk_size'], args['action_dim']).cuda()
    
    # 1. Training Forward
    pred_actions, mu, log_var = model(dummy_qpos, dummy_image, dummy_actions)
    print(f"Training Output Shape: {pred_actions.shape}") # Should be [2, 20, 7]
    
    # 2. Inference Forward (No actions provided)
    model.eval()
    with torch.no_grad():
        pred_actions_inf, _, _ = model(dummy_qpos, dummy_image)
    print(f"Inference Output Shape: {pred_actions_inf.shape}")