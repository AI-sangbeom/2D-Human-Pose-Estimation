# Few-Shot Keypoint Detection Implementation Plan
## Based on DINOv3 Small Backbone

## Core Concept
- **Backbone**: DINOv3 Small (patch_size=16, embed_dim=384, depth=12, num_heads=6)
- **Support**: N-way K-shot keypoint examples
- **Query**: Images to predict keypoints
- **Goal**: Learn meta-knowledge for rapid adaptation to new keypoint patterns

## Implementation Plan

### Phase 1: Enhanced FSKD Model Architecture
1. **Meta-Learning Capable FSKD**
   - Support/query feature extraction
   - Prototypical-based prediction
   - Episode-based training support

2. **DINOv3 Small Integration**
   - Use existing Dinov3ViT small configuration
   - Multi-scale feature extraction
   - Fine-tuning capabilities

### Phase 2: Few-Shot Learning Components
1. **Prototypical Networks for Keypoints**
   - Compute prototypes from support examples
   - Distance-based classification/regression
   - Few-shot inference mechanism

2. **Attention-Based Feature Fusion**
   - Support-query attention mechanism
   - Multi-head attention for keypoint alignment
   - Feature aggregation strategies

### Phase 3: Episodic Training Framework
1. **N-way K-shot Episodes**
   - Support set: K examples per class/subject
   - Query set: Images to predict
   - Meta-training and meta-validation

2. **Training Objectives**
   - Prototypical loss
   - Keypoint regression loss
   - Meta-learning regularization

### Phase 4: Model Components
```
models/few_shot_fskd/
├── __init__.py
├── fskd_small.py              # Main FSKD with DINOv3 small
├── prototypical_head.py       # Prototypical prediction head
├── attention_fusion.py        # Support-query attention
├── feature_extractor.py       # DINOv3 small feature extraction
└── meta_classifier.py         # Meta-learning classifier

data/
├── few_shot_dataset.py        # Few-shot dataset wrapper
├── episode_generator.py       # Episode generation
└── few_shot_loader.py        # Episodic data loader

engine/
├── few_shot_trainer.py       # Meta-learning trainer
├── episodic_evaluator.py     # Few-shot evaluation
└── meta_optimizer.py         # Meta-optimizer
```

## Technical Implementation Details

### 1. Enhanced FSKD Model (fskd_small.py)
```python
class FewShotFSKD(nn.Module):
    def __init__(self, nkpts: int, backbone: str = 'small', pretrained: bool = True):
        super().__init__()
        backbone_cfg = vit_sizes[backbone]  # Use DINOv3 small
        embed_dim = backbone_cfg['embed_dim']  # 384 for small
        
        # DINOv3 Small backbone
        self.backbone = Dinov3ViT(
            patch_size=backbone_cfg["patch_size"],
            embed_dim=embed_dim,
            depth=backbone_cfg["depth"],
            num_heads=backbone_cfg["num_heads"],
            ffn_ratio=backbone_cfg["ffn_ratio"],
            pretrained=pretrained,
        )
        
        # Feature extraction layers
        self.feature_extractor = FeatureExtractor(embed_dim)
        self.prototypical_head = PrototypicalHead(nkpts, embed_dim)
        self.attention_fusion = AttentionFusion(embed_dim)
        
    def forward(self, support_imgs, support_keypoints, query_imgs):
        # Extract features
        support_features = self.feature_extractor(support_imgs)
        query_features = self.feature_extractor(query_imgs)
        
        # Compute prototypes
        prototypes = self.prototypical_head(support_features, support_keypoints)
        
        # Fuse with query features
        fused_features = self.attention_fusion(prototypes, query_features)
        
        # Predict keypoints
        predictions = self.prototypical_head.predict_keypoints(fused_features)
        
        return predictions
```

### 2. Key Components

#### Feature Extractor
- Use DINOv3 small backbone for feature extraction
- Multi-scale feature representation
- Feature normalization and projection

#### Prototypical Head
- Compute class/subject prototypes from support examples
- Distance-based keypoint prediction
- Support for N-way K-shot scenarios

#### Attention Fusion
- Cross-attention between support prototypes and query features
- Learnable attention weights
- Feature alignment mechanism

### 3. Training Pipeline

#### Episodic Data Loading
```python
class FewShotEpisode:
    def __init__(self, N_way, K_shot, num_queries):
        self.N_way = N_way
        self.K_shot = K_shot  
        self.num_queries = num_queries
        
    def sample_episode(self, dataset):
        # Sample N_way classes
        classes = dataset.sample_classes(self.N_way)
        
        # Sample K_shot support examples per class
        support_set = []
        for cls in classes:
            support_set.extend(dataset.sample_examples(cls, self.K_shot))
            
        # Sample query examples
        query_set = []
        for cls in classes:
            query_set.extend(dataset.sample_examples(cls, self.num_queries))
            
        return support_set, query_set
```

#### Training Loop
```python
def train_episode(model, optimizer, support_batch, query_batch):
    # Forward pass
    predictions = model(support_imgs, support_keypoints, query_imgs)
    
    # Compute loss
    loss = prototypical_loss(predictions, query_keypoints)
    
    # Meta-update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

## Configuration
```yaml
# configs/few_shot/fskd_small.yaml
model:
  name: FewShotFSKD
  backbone: small  # DINOv3 small
  nkpts: 17
  pretrained: true

train:
  n_way: 5
  k_shot: 1
  num_queries: 1
  meta_lr: 1e-3
  num_episodes: 10000
  
data:
  dataset: MPII
  episode_length: 100
  meta_batch_size: 4
```

## Expected Benefits
1. **Efficient Feature Extraction**: DINOv3 small provides strong visual representations
2. **Rapid Adaptation**: Few-shot learning enables quick adaptation to new subjects
3. **Scalable Architecture**: Support for various N-way K-shot configurations
4. **Strong Generalization**: Meta-learning improves generalization across subjects

## Implementation Priority
1. Implement FewShotFSKD model with DINOv3 small backbone
2. Add prototypical head and attention fusion
3. Create episodic data loader
4. Implement meta-training pipeline
5. Add evaluation protocol for few-shot scenarios
