import math
from matplotlib.pyplot import axis
import torch
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
from transformer import ViT
from einops import rearrange, repeat


def setup_seed(seed):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dot_product_decode(Z):
    A_pred = F.leaky_relu(torch.matmul(Z, Z.t()))
    return A_pred


class Multi_LSTM(nn.Module):
    def __init__(self,
                 emb_d,
                 emb_attri,
                 n_aeunits,
                 class_num,
                 use_attribute=True,
                 dropout=0.5):
        super(Multi_LSTM, self).__init__()
        if use_attribute:
            self.rnn1 = nn.LSTM(input_size=emb_d + emb_attri,
                                hidden_size=64,
                                num_layers=2,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=False)
        else:
            self.rnn1 = nn.LSTM(input_size=emb_d,
                                hidden_size=64,
                                num_layers=2,
                                dropout=dropout,
                                batch_first=True,
                                bidirectional=False)

        self.classifier = nn.Linear(64, class_num)

    def forward(self, x):
        out, _ = self.rnn1(x)
        return self.classifier(out[:, -1, :]),out[:, -1, :]


class BiLSTM_Attention(nn.Module):
    def __init__(self, embedding_dim, n_hidden, num_classes, dropout_n=0.5):
        super(BiLSTM_Attention, self).__init__()

        self.lstm = nn.LSTM(embedding_dim,
                            n_hidden,
                            dropout=dropout_n,
                            bidirectional=True)
        self.out = nn.Linear(n_hidden * 2, num_classes)

    def attention_net(self, lstm_output, final_state):
        batch_size = len(lstm_output)
        hidden = final_state.view(batch_size, -1, 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)

        context = torch.bmm(lstm_output.transpose(1, 2),
                            soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights

    def forward(self, X):
        '''
        X: [batch_size, seq_len]
        '''
        input = X
        input = input.transpose(0, 1)
        output, (final_hidden_state, final_cell_state) = self.lstm(input)
        output = output.transpose(0, 1)
        attn_output, attention = self.attention_net(output, final_hidden_state)
        return self.out(attn_output), attention
        


class SelfAttention(nn.Module):
    def __init__(self, dropout, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = torch.softmax(scores, dim=2)
        return torch.bmm(self.dropout(self.attention_weights), values)


class SpatioTemporalAttention(nn.Module):
    """
    Adaptive Spatio-Temporal Encoder (ASTE) module from the paper.
    Implements dual-stream attention with adaptive fusion (Eq. 5).
    """
    
    def __init__(self, emb_d, num_heads=4, dropout=0.1, use_temporal_stream=True, use_spatial_stream=True):
        super(SpatioTemporalAttention, self).__init__()
        self.emb_d = emb_d
        self.num_heads = num_heads
        self.head_dim = emb_d // num_heads
        self.use_temporal_stream = use_temporal_stream  
        self.use_spatial_stream = use_spatial_stream  
        
        assert emb_d % num_heads == 0, "emb_d must be divisible by num_heads"
        
        self.temporal_q = nn.Linear(emb_d, emb_d)
        self.temporal_k = nn.Linear(emb_d, emb_d)
        self.temporal_v = nn.Linear(emb_d, emb_d)
        
        self.spatial_q = nn.Linear(emb_d, emb_d)
        self.spatial_k = nn.Linear(emb_d, emb_d)
        self.spatial_v = nn.Linear(emb_d, emb_d)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_d * 2, emb_d),
            nn.ReLU(),
            nn.Linear(emb_d, 2),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_d)
        
    def forward(self, x):
        
        N, T, D = x.shape
        residual = x
        
        t_out = None
        s_out = None
        
        if self.use_temporal_stream:
            t_q = self.temporal_q(x)
            t_k = self.temporal_k(x)
            t_v = self.temporal_v(x)
            
            t_q = t_q.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
            t_k = t_k.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
            t_v = t_v.view(N, T, self.num_heads, self.head_dim).transpose(1, 2)
            
            t_scores = torch.matmul(t_q, t_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            t_attn = F.softmax(t_scores, dim=-1)
            t_attn = self.dropout(t_attn)
            t_out = torch.matmul(t_attn, t_v)
            t_out = t_out.transpose(1, 2).contiguous().view(N, T, D)
        
        if self.use_spatial_stream:
            x_temporal_mean = x.mean(dim=1)
            
            s_context = self.spatial_q(x_temporal_mean)
            s_context_expanded = s_context.unsqueeze(1).expand(N, T, D)
            s_out = 0.5 * x + 0.5 * s_context_expanded
        
        if self.use_temporal_stream and self.use_spatial_stream:
            fused = torch.cat([t_out, s_out], dim=-1)
            gate_weights = self.fusion_gate(fused)
            out = gate_weights[..., 0:1] * t_out + gate_weights[..., 1:2] * s_out
        elif self.use_temporal_stream:
            out = t_out
        elif self.use_spatial_stream:
            out = s_out
        else:
            out = x
        
        out = self.layer_norm(out + residual)
        
        return out


class RwR_NN(nn.Module):
    def __init__(
            self,
            n_node,
            prob_c=0.98,  
            step_k=5,  
            t=10,
            attri_emd=100,
            dropout=0.5
            ):
        super(RwR_NN, self).__init__()
        self.look_back = t
        self.step_k = step_k
        self.prob_c = prob_c
        self.weight = Parameter(data=torch.ones(t, step_k) * prob_c,requires_grad=True)
        self.Pt0 = Parameter(data=torch.eye(n_node),
                             requires_grad=False)  
        self.LN = nn.LayerNorm(n_node)
        self.BN = nn.BatchNorm1d(n_node)
    def get_alpha(self):
        return self.weight.data.cpu().data.numpy()

    def forward(self, input):
        """
        CRITICAL FIX: Memory-efficient time-slicing version to avoid OOM
        Instead of accumulating all timesteps in a list and stacking at the end,
        we pre-allocate the output tensor and compute timestep-by-timestep.
        """
        w = self.weight.data
        w = w.clamp(0, 1)
        self.weight.data = w
        n_time, n_node = input.shape[1], input.shape[0]
        
        # Pre-allocate output tensor to avoid memory fragmentation from list operations
        output_tensor = torch.zeros(n_node, n_time, n_node, device=input.device, dtype=input.dtype)
        
        # Compute timestep-by-timestep, releasing intermediate variables immediately
        for t in range(n_time):
            Gt = input[:, t, :]
            
            # Add small epsilon to avoid division by zero
            degree_sum = torch.sum(Gt, axis=1) + 1e-8
            Dt_inv = torch.diag(1 / degree_sum)
            Gt_norm = torch.mm(Dt_inv, Gt)
            
            Pt_p = self.Pt0.to(input.device)  # Ensure on same device
            
            # Multi-step random walk
            final_P = None
            for i in range(self.step_k):
                Pt_c = self.weight[t, i] * torch.mm(Pt_p, Gt_norm) + (
                    1 - self.weight[t, i]) * self.Pt0.to(input.device)
                
                if i == 0:
                    final_P = Pt_c.clone()
                else:
                    final_P = final_P + Pt_c
                Pt_p = Pt_c
            
            # Normalize and store result, then immediately release intermediate variables
            output_tensor[:, t, :] = self.LN(final_P)
            
            # Explicitly delete intermediate tensors to free memory faster
            del Gt, degree_sum, Dt_inv, Gt_norm, Pt_p, final_P, Pt_c
        
        return output_tensor

class AutoEnmodel(nn.Module):
    def __init__(self,
                 node_numb,
                 emb_d,
                 n_aeunits=[500, 300],
                 t=10,
                 class_num=5,
                 attri_emd=100,
                 dropout=0.5,
                 use_attribute=True,
                 use_rwr=False,
                 use_Transformer=False,
                 use_knowledge=False,  
                 knowledge_dim=64,
                 use_tcl=False,  
                 tcl_weight=0.02,
                 contrastive_tau=0.5,
                 use_multiscale=False,  
                 multiscale_windows=[4],
                 transformer_layers=6,  
                 attention_heads=8,     
                 multiscale_transformer_layers=4,  
                 multiscale_attention_heads=4,     
                 use_spatiotemporal_attention=True,  
                 use_temporal_stream=True,  
                 use_spatial_stream=True,  
                 use_adaptive_kem=False,  
                 use_sparsity_aware_attention=False,  
):
        super(AutoEnmodel, self).__init__()
        self.look_back = t
        self.use_attribute = use_attribute
        self.use_rwr = use_rwr
        self.use_knowledge = use_knowledge
        self.use_tcl = use_tcl
        self.tcl_weight = tcl_weight
        self.contrastive_tau = contrastive_tau
        self.use_multiscale = use_multiscale
        self.multiscale_windows = multiscale_windows  
        # Filter multiscale windows to only include valid ones (window <= time_steps)
        self.valid_multiscale_windows = [w for w in self.multiscale_windows if w <= t] if self.multiscale_windows else []
        self.node_numb = node_numb
        self.use_temporal_stream = use_temporal_stream  
        self.use_spatial_stream = use_spatial_stream  
        self.use_adaptive_kem = use_adaptive_kem  
        self.use_sparsity_aware_attention = use_sparsity_aware_attention  

        self.class_num = class_num
        self.embedding = torch.zeros([node_numb,emb_d + attri_emd ])
        self.rwr = RwR_NN(node_numb)
        enmodel = [
            nn.Sequential(
                nn.Linear(node_numb, n_aeunits[0]),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(n_aeunits[0], n_aeunits[1]),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(n_aeunits[1], emb_d)
            ) for i in range(self.look_back)
        ]
        self.enmodel = nn.ModuleList(enmodel)

        self.classifier = nn.Linear(emb_d + attri_emd, class_num)
        self.ln = nn.LayerNorm([t, node_numb])

        if use_attribute:
            attri_DNN = [nn.Sequential(
                nn.Linear(attri_emd, emb_d),
                nn.Dropout(dropout),
            )for i in range(self.look_back)]
            self.attri_DNN = nn.ModuleList(attri_DNN)
        self.atten = nn.Linear(t, t)
        self.knowledge_dim = knowledge_dim
        if self.use_knowledge:
            self.knowledge_emb = nn.Embedding(node_numb, knowledge_dim)
            nn.init.xavier_uniform_(self.knowledge_emb.weight)
            # Multi-layer perceptron for knowledge projection (Eq. 3 in paper)
            self.knowledge_proj = nn.Sequential(
                nn.Linear(knowledge_dim, emb_d * 2),
                nn.ReLU(),
                nn.Linear(emb_d * 2, emb_d)
            )
            
            self.knowledge_norm = nn.LayerNorm(knowledge_dim)
            self.knowledge_dropout = nn.Dropout(0.1)
            
            self.register_buffer('node_degrees', torch.zeros(node_numb))
            self.register_buffer('knowledge_initialized', torch.tensor(False))
            self.fusion_gate = nn.Sequential(
                nn.Linear(emb_d * 3, emb_d),
                nn.LeakyReLU(),
                nn.Linear(emb_d, 3)
            )
          
            self.knowledge_scale = nn.Parameter(torch.tensor(-2.0))  # Balanced scale for RWR features
            self.alpha_knowledge = getattr(self, 'alpha_knowledge', 0.1)  
            self.alpha_structure = getattr(self, 'alpha_structure', 0.6)  
            self.alpha_attribute = getattr(self, 'alpha_attribute', 0.3)  
            self.register_buffer('gate_eps', torch.tensor(1e-8))
            
            if self.use_adaptive_kem:
                self.sparsity_encoder = nn.Sequential(
                    nn.Linear(1, 32),
                    nn.ReLU(),
                    nn.Linear(32, 64),
                    nn.ReLU()
                )
              
                gate_out_layer = nn.Linear(64, 1)
                nn.init.constant_(gate_out_layer.bias, -0.8)
                nn.init.xavier_normal_(gate_out_layer.weight, gain=0.1)
                
                self.adaptive_gate_network = nn.Sequential(
                    nn.Linear(64 + emb_d, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    gate_out_layer,
                    nn.Sigmoid()
                )
                self.register_buffer('gate_history', torch.zeros(100))
                self.register_buffer('gate_history_idx', torch.tensor(0))
        else:
           
            self.fusion_gate = nn.Sequential(
                nn.Linear(emb_d * (2 if use_attribute else 1), emb_d),
                nn.LeakyReLU(),
                nn.Linear(emb_d, 2 if use_attribute else 1)
            )
        
        self.use_spatiotemporal_attention = use_spatiotemporal_attention
        if self.use_spatiotemporal_attention:
           
            sta_heads = min(4, attention_heads // 2)
            self.spatiotemporal_attn = SpatioTemporalAttention(
                emb_d=emb_d,
                num_heads=sta_heads,
                dropout=min(0.2, dropout * 0.6),  
                use_temporal_stream=use_temporal_stream,  
                use_spatial_stream=use_spatial_stream  
            )
        
        if use_Transformer:
            vit_dim = emb_d
        

            if self.use_multiscale and use_Transformer:
                # Use valid multiscale windows for multi-resolution temporal modeling
                self.multiscale_encoders = nn.ModuleList()
                for window in self.valid_multiscale_windows:
                    encoder = ViT(time=window,
                                dim=vit_dim,
                                num_classes=vit_dim,
                                depth=2,  
                                heads=multiscale_attention_heads,
                                mlp_dim=128,
                                pool='mean',
                                dropout=dropout,
                                emb_dropout=dropout,
                                pos=False)
                    self.multiscale_encoders.append(encoder)
                

                fusion_input_dim = vit_dim * (len(self.valid_multiscale_windows) + 1)
                self.multiscale_fusion = nn.Sequential(
                    nn.Linear(fusion_input_dim, vit_dim),
                    nn.LayerNorm(vit_dim),
                    nn.Dropout(dropout * 0.8),  
                    nn.Linear(vit_dim, vit_dim)
                )
            
            self.lstm = ViT(time=self.look_back,
                            dim=vit_dim,
                            num_classes=self.class_num,
                            depth=transformer_layers,  
                            heads=attention_heads,      
                            mlp_dim=256,   
                            pool='mean',
                            dropout=dropout,
                            emb_dropout=dropout,
                            pos=False)

        else:
        
            self.lstm = Multi_LSTM(emb_d, 0, n_aeunits, class_num, False,
                                   dropout)

  
    def get_embedding(self):
        return self.embedding.cpu().detach().numpy()
    
    def init_knowledge_from_graph(self, adj_matrices, pagerank_scores=None, clustering_coeff=None, turnover_rates=None):
   
        if not self.use_knowledge or self.knowledge_initialized:
            return
            
        device = next(self.parameters()).device
        node_features = []
        
        for adj in adj_matrices:
            if adj is not None:

                if isinstance(adj, np.ndarray):
                    adj_tensor = torch.from_numpy(adj).float().to(device)
                else:
                    adj_tensor = adj.to(device)
                

                degrees = torch.sum(adj_tensor, dim=-1) + torch.sum(adj_tensor, dim=-2)
                node_features.append(degrees)
        
        if len(node_features) > 0:

            avg_degrees = torch.stack(node_features, dim=0).mean(dim=0)
            
            if avg_degrees.max() > 0:
                normalized_degrees = (avg_degrees - avg_degrees.mean()) / (avg_degrees.std() + 1e-8)
            else:
                normalized_degrees = avg_degrees
            
            # FIX: Use normalized raw degrees for squaring to preserve monotonicity
            # Paper Eq.1 requires degree^2 to be monotonically increasing
            max_deg = avg_degrees.max() + 1e-6
            
            base_features = [
                normalized_degrees, 
                torch.log(avg_degrees + 1),  
                torch.sqrt(avg_degrees + 1),  
                (avg_degrees / max_deg) ** 2,  # Fixed: preserve monotonicity for structural prior
            ]
            

            if pagerank_scores is not None:
                pr_tensor = torch.from_numpy(pagerank_scores).float().to(device)
                pr_normalized = (pr_tensor - pr_tensor.mean()) / (pr_tensor.std() + 1e-8)
                base_features.append(pr_normalized)
            
            if clustering_coeff is not None:
                cc_tensor = torch.from_numpy(clustering_coeff).float().to(device)
                cc_normalized = (cc_tensor - cc_tensor.mean()) / (cc_tensor.std() + 1e-8)
                base_features.append(cc_normalized)
            
            if turnover_rates is not None:
                tr_tensor = torch.from_numpy(turnover_rates).float().to(device)
                tr_normalized = (tr_tensor - tr_tensor.mean()) / (tr_tensor.std() + 1e-8)
                base_features.append(tr_normalized)
            

            # CRITICAL FIX: Avoid random projection for reproducibility
            # Previous bug: created new nn.Linear each time, causing non-deterministic initialization
            if self.knowledge_dim > len(base_features):
                knowledge_features = torch.stack(base_features, dim=1)
                # Deterministic expansion: repeat last feature or zero-pad instead of random projection
                n_nodes = knowledge_features.shape[0]
                n_existing = knowledge_features.shape[1]
                n_needed = self.knowledge_dim - n_existing
                # Zero-padding for deterministic behavior
                padding = torch.zeros(n_nodes, n_needed, device=device)
                knowledge_features = torch.cat([knowledge_features, padding], dim=1)
            else:
                knowledge_features = torch.stack(base_features[:self.knowledge_dim], dim=1)
            

            with torch.no_grad():
                self.knowledge_emb.weight.data = knowledge_features.to(device)
                self.knowledge_initialized.fill_(True)
    
    def attention(self, adj, attri):

        n, t, _ = adj.shape


        if self.use_attribute and attri is not None:
            attri_emb_with_time = []
            for ti, layer in enumerate(self.attri_DNN):
                attri_emb_with_time.append(layer(attri[:, ti, :]))
            attri_emb = torch.stack(attri_emb_with_time, dim=1) 
        else:
            attri_emb = torch.zeros_like(adj)

       
        if self.use_knowledge:
            device = adj.device
            node_idx = torch.arange(self.node_numb, device=device)
            k0 = self.knowledge_emb(node_idx)                      
            k0 = self.knowledge_norm(k0)
            k0 = self.knowledge_dropout(k0)
            k = self.knowledge_proj(k0)                              
            k = k.unsqueeze(1).expand(n, t, k.shape[-1])             
        else:
            k = torch.zeros_like(adj)

        if self.use_knowledge:

            if self.use_adaptive_kem and hasattr(self, 'adaptive_gate_network'):
                
                if hasattr(self, 'current_adj_matrix') and self.current_adj_matrix is not None:
                    avg_degrees = self.current_adj_matrix.sum(dim=-1).mean(dim=1)  
                    isolated_ratio = (avg_degrees == 0).float().mean()
                    sparsity = isolated_ratio.unsqueeze(0).unsqueeze(0)  
                else:
                    sparsity = torch.tensor([[0.5]], device=adj.device)
                

                sparsity_emb = self.sparsity_encoder(sparsity)  
                sparsity_emb = sparsity_emb.expand(n, -1)  
                
             
                node_repr = adj.mean(dim=1)  
                gate_input = torch.cat([sparsity_emb, node_repr], dim=-1) 
                adaptive_gate = self.adaptive_gate_network(gate_input)  
                

                if self.training:
                    idx = self.gate_history_idx.item() % 100
                    self.gate_history[idx] = adaptive_gate.mean().item()
                    self.gate_history_idx += 1
                

                adaptive_gate = adaptive_gate.unsqueeze(1).expand(n, t, 1)  
                

                base_features = 0.7 * adj + 0.3 * attri_emb

                out = (1 - adaptive_gate) * base_features + adaptive_gate * k

            # CRITICAL FIX: Comment out fixed-weight branch to force adaptive gating (Eq.5)
            # The existence of alpha_structure would lock model into fixed weights (0.6, 0.3, 0.1)
            # This prevents the adaptive Softmax gating from being used
            # elif hasattr(self, 'alpha_structure') and hasattr(self, 'alpha_attribute') and hasattr(self, 'alpha_knowledge'):
            #     alpha_s = self.alpha_structure
            #     alpha_a = self.alpha_attribute  
            #     alpha_k = self.alpha_knowledge
            #     total = alpha_s + alpha_a + alpha_k
            #     if total > 0:
            #         alpha_s, alpha_a, alpha_k = alpha_s/total, alpha_a/total, alpha_k/total
            #     else:
            #         alpha_s, alpha_a, alpha_k = 0.6, 0.3, 0.1
            #     
            #     out = alpha_s * adj + alpha_a * attri_emb + alpha_k * k
            else:
                # FORCED: Now always use adaptive Softmax gating (Paper Eq.5)
               
                fused_in = torch.cat([adj, attri_emb, k], dim=2)        
                gate_logits = self.fusion_gate(fused_in)                 
                weights = F.softmax(gate_logits, dim=-1)
               
                k_scale = torch.sigmoid(self.knowledge_scale)            
                scaled = torch.stack([weights[..., 0],
                                      weights[..., 1],
                                      k_scale * weights[..., 2]], dim=-1)  
                denom = torch.sum(scaled, dim=-1, keepdim=True) + self.gate_eps
                weights = scaled / denom
                out = (
                    weights[..., 0:1] * adj +
                    weights[..., 1:2] * attri_emb +
                    weights[..., 2:3] * k
                )
        else:
            if self.use_attribute:
                fused_in = torch.cat([adj, attri_emb], dim=2)        
                gate_logits = self.fusion_gate(fused_in)             
                weights = F.softmax(gate_logits, dim=-1)
                out = weights[..., 0:1] * adj + weights[..., 1:2] * attri_emb
            else:
                out = adj
        return out

    def load_external_knowledge(self, knowledge_matrix: torch.Tensor, freeze: bool = True):
        if not self.use_knowledge:
            return
        if knowledge_matrix.shape != (self.node_numb, self.knowledge_dim):
            raise ValueError('knowledge_matrix shape mismatch')
        with torch.no_grad():
            self.knowledge_emb.weight.copy_(knowledge_matrix.to(self.knowledge_emb.weight.device))
        self.knowledge_emb.weight.requires_grad_(not freeze)
    
    def set_ablation_weights(self, alpha_structure=0.6, alpha_attribute=0.3, alpha_knowledge=0.1):
        if self.use_knowledge:
            self.alpha_structure = alpha_structure
            self.alpha_attribute = alpha_attribute
            self.alpha_knowledge = alpha_knowledge
    
    def set_adjacency_matrix(self, adj_matrix):
        
        self.current_adj_matrix = adj_matrix
    
    def get_adaptive_gate_statistics(self):
        
        if not self.use_adaptive_kem or not hasattr(self, 'gate_history'):
            return None
        
       
        valid_gates = self.gate_history[self.gate_history != 0]
        if len(valid_gates) == 0:
            return None
        
        stats = {
            'mean': valid_gates.mean().item(),
            'std': valid_gates.std().item(),
            'min': valid_gates.min().item(),
            'max': valid_gates.max().item(),
            'history': valid_gates.cpu().numpy()
        }
        return stats

    def forward(self, x, y=None):   
       
        if self.use_rwr:
            x = self.rwr(x)
        

        encoded_with_time = []        
        for t, layer in enumerate(self.enmodel):
            encoded_with_time.append(layer(x[:, t, :]))
        encoded = torch.stack(encoded_with_time, dim=1)

        if self.use_attribute:          
            encoded = self.attention(encoded, y)

       
        if hasattr(self, 'use_spatiotemporal_attention') and self.use_spatiotemporal_attention:
            encoded = self.spatiotemporal_attn(encoded) 


        if self.use_multiscale and hasattr(self, 'multiscale_encoders'):
            encoded_copy = encoded.clone()
            multiscale_features = []
            
            if encoded_copy.shape[1] >= self.look_back:
                original_features = encoded_copy[:, :self.look_back, :].clone()  
                multiscale_features.append(original_features.mean(dim=1))  
            

            # Use all valid multiscale windows as defined in initialization
            for i, (encoder, window) in enumerate(zip(self.multiscale_encoders, self.valid_multiscale_windows)):
                if encoded_copy.shape[1] >= window:
                    windowed_features = encoded_copy[:, -window:, :].clone()  
                    scale_output, scale_features = encoder(windowed_features)
                    multiscale_features.append(scale_features)  
            
            if len(multiscale_features) > 1:
                fused_features = torch.cat(multiscale_features, dim=-1)  
                encoded_enhanced = self.multiscale_fusion(fused_features)  
                if encoded_copy.shape[1] >= self.look_back:
                    encoded_enhanced = encoded_enhanced + encoded_copy[:, :self.look_back, :].mean(dim=1)
                encoded = encoded_enhanced.unsqueeze(1).repeat(1, encoded.shape[1], 1)

        
        self.contrastive_loss = torch.tensor(0.0, device=encoded.device)
        if self.use_tcl and encoded.shape[1] > 1:
            encoded_tcl = encoded.clone().contiguous()          
            losses = []
            tau = self.contrastive_tau
    
            for ti in range(1, min(encoded_tcl.shape[1], 3)):  
                z_q = F.normalize(encoded_tcl[:, ti, :].contiguous(), dim=-1)      
                z_k = F.normalize(encoded_tcl[:, ti - 1, :].contiguous(), dim=-1)  
                logits = torch.matmul(z_q, z_k.t()) / tau                          
                labels = torch.arange(z_q.shape[0], device=encoded.device)
                losses.append(F.cross_entropy(logits, labels))
            if len(losses) > 0:
                n_nodes = encoded_tcl.shape[0]
                reduced_weight = self.tcl_weight * 0.5
                scale = reduced_weight / max(1.0, math.log(max(2, n_nodes)))
                self.contrastive_loss = scale * torch.stack(losses).mean()


        # CRITICAL FIX: Return (features, logits) not (encoded, encoded)
        # Training code expects: features, output = model(...)
        # where features are used for contrastive learning and output for classification
        logits, feats = self.lstm(encoded)
        self.embedding = feats
        return feats, logits  # (features for contrastive, logits for classification)



class Weighted_mse_x(nn.Module):
    def __init__(self, beta):
        super(Weighted_mse_x, self).__init__()
        self.beta = beta

    def forward(self, y_pred, lable):
        
        y_batch = torch.ones_like(lable)  
        y_batch[lable != 0] = self.beta
        y_batch[lable == 0] = -1
        loss = torch.sum(torch.square(y_batch * torch.sub(y_pred, lable)),
                         axis=-1)
        return torch.sum(loss)
