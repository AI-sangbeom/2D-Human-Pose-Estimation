import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import math


class FewShotDataset(Dataset):
    """
    Few-shot dataset wrapper that handles episodic sampling
    Converts standard dataset into few-shot learning format
    """
    
    def __init__(
        self,
        base_dataset,
        n_way: int = 5,
        k_shot: int = 1,
        n_queries: int = 1,
        classes_per_episode: Optional[List] = None,
        class_sampling: str = 'random',  # 'random', 'balanced', 'fixed'
        transform=None,
        return_metadata: bool = True
    ):
        """
        Initialize few-shot dataset
        
        Args:
            base_dataset: Original dataset with images and annotations
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_queries: Number of query examples per class
            classes_per_episode: Fixed classes for each episode (optional)
            class_sampling: How to sample classes
            transform: Image transformations
            return_metadata: Whether to return metadata
        """
        self.base_dataset = base_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_queries = n_queries
        self.classes_per_episode = classes_per_episode
        self.class_sampling = class_sampling
        self.transform = transform
        self.return_metadata = return_metadata
        
        # Analyze base dataset structure
        self._analyze_dataset()
        
        # Class-to-samples mapping
        self.class_to_samples = self._build_class_mapping()
        
        # Predefined episodes if using fixed sampling
        if classes_per_episode is not None:
            self.episodes = self._generate_fixed_episodes(classes_per_episode)
        else:
            self.episodes = None
    
    def _analyze_dataset(self):
        """Analyze the base dataset to understand its structure"""
        # This would depend on the specific dataset format
        # For now, assume dataset returns (image, keypoints, metadata)
        if hasattr(self.base_dataset, '__len__'):
            self.total_samples = len(self.base_dataset)
        else:
            self.total_samples = 1000  # Default assumption
            
        # Extract class information
        self.classes = self._extract_classes()
        self.num_classes = len(self.classes)
        
        print(f"Dataset analysis: {self.total_samples} samples, {self.num_classes} classes")
    
    def _extract_classes(self) -> List:
        """Extract class information from dataset"""
        # This would depend on the specific dataset
        # For MPII dataset, classes might be activity categories
        # For general keypoint datasets, might be subject IDs
        classes = []
        
        # Try to extract from dataset metadata
        if hasattr(self.base_dataset, 'classes'):
            classes = self.base_dataset.classes
        elif hasattr(self.base_dataset, 'get_classes'):
            classes = self.base_dataset.get_classes()
        else:
            # Default: assume sequential class IDs
            classes = list(range(self.num_classes))
        
        return classes
    
    def _build_class_mapping(self) -> Dict:
        """Build mapping from classes to sample indices"""
        class_to_samples = defaultdict(list)
        
        # This would iterate through dataset to find samples for each class
        # Implementation depends on dataset format
        for idx in range(len(self.base_dataset)):
            try:
                sample = self.base_dataset[idx]
                class_label = self._extract_class_label(sample)
                class_to_samples[class_label].append(idx)
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
        
        return dict(class_to_samples)
    
    def _extract_class_label(self, sample) -> int:
        """Extract class label from a sample"""
        # This depends on dataset format
        # Assume sample is (image, keypoints, metadata)
        if len(sample) >= 3:
            metadata = sample[2]
            if isinstance(metadata, dict) and 'class' in metadata:
                return metadata['class']
            elif isinstance(metadata, (int, str)):
                return metadata
        
        # Default: use index as class
        return 0
    
    def _generate_fixed_episodes(self, classes_per_episode: List[List]) -> List:
        """Generate episodes with fixed class combinations"""
        episodes = []
        
        for episode_classes in classes_per_episode:
            if len(episode_classes) != self.n_way:
                raise ValueError(f"Expected {self.n_way} classes, got {len(episode_classes)}")
            
            episode = {
                'support': {},
                'query': {},
                'classes': episode_classes
            }
            
            # Sample support and query for each class
            for cls in episode_classes:
                samples = self.class_to_samples.get(cls, [])
                
                if len(samples) < self.k_shot + self.n_queries:
                    print(f"Warning: Not enough samples for class {cls}")
                    continue
                
                # Randomly sample support and query
                sampled = random.sample(samples, self.k_shot + self.n_queries)
                episode['support'][cls] = sampled[:self.k_shot]
                episode['query'][cls] = sampled[self.k_shot:self.k_shot + self.n_queries]
            
            episodes.append(episode)
        
        return episodes
    
    def sample_episode(self) -> Dict:
        """Sample a new episode"""
        if self.episodes is not None:
            # Return predefined episode
            return random.choice(self.episodes)
        
        # Sample classes
        if self.class_sampling == 'random':
            episode_classes = random.sample(self.classes, self.n_way)
        elif self.class_sampling == 'balanced':
            # Sample classes with balanced representation
            episode_classes = random.sample(self.classes, self.n_way)
        else:
            episode_classes = random.sample(self.classes, self.n_way)
        
        # Build episode
        episode = {
            'support': {},
            'query': {},
            'classes': episode_classes
        }
        
        # Sample support and query for each class
        for cls in episode_classes:
            samples = self.class_to_samples.get(cls, [])
            
            if len(samples) < self.k_shot + self.n_queries:
                # Fallback: use available samples
                available = len(samples)
                k_shot = min(self.k_shot, available)
                n_queries = min(self.n_queries, max(0, available - k_shot))
            else:
                k_shot = self.k_shot
                n_queries = self.n_queries
            
            if available > 0:
                sampled = random.sample(samples, k_shot + n_queries)
                episode['support'][cls] = sampled[:k_shot]
                episode['query'][cls] = sampled[k_shot:k_shot + n_queries]
        
        return episode
    
    def __len__(self):
        if self.episodes is not None:
            return len(self.episodes)
        else:
            return 1000  # Infinite episodes for random sampling
    
    def __getitem__(self, idx):
        """Get a single episode"""
        episode = self.sample_episode()
        
        # Process support set
        support_images = []
        support_keypoints = []
        support_metadata = []
        
        for cls in episode['classes']:
            for sample_idx in episode['support'].get(cls, []):
                sample = self.base_dataset[sample_idx]
                image, keypoints, metadata = self._process_sample(sample)
                
                support_images.append(image)
                support_keypoints.append(keypoints)
                support_metadata.append({**metadata, 'class': cls, 'role': 'support'})
        
        # Process query set
        query_images = []
        query_keypoints = []
        query_metadata = []
        
        for cls in episode['classes']:
            for sample_idx in episode['query'].get(cls, []):
                sample = self.base_dataset[sample_idx]
                image, keypoints, metadata = self._process_sample(sample)
                
                query_images.append(image)
                query_keypoints.append(keypoints)
                query_metadata.append({**metadata, 'class': cls, 'role': 'query'})
        
        # Stack tensors
        support_images = torch.stack(support_images) if support_images else torch.empty(0)
        support_keypoints = torch.stack(support_keypoints) if support_keypoints else torch.empty(0)
        query_images = torch.stack(query_images) if query_images else torch.empty(0)
        query_keypoints = torch.stack(query_keypoints) if query_keypoints else torch.empty(0)
        
        # Compile episode data
        episode_data = {
            'support_images': support_images,
            'support_keypoints': support_keypoints,
            'query_images': query_images,
            'query_keypoints': query_keypoints,
            'n_way': len(episode['classes']),
            'k_shot': len(support_images) // len(episode['classes']) if episode['classes'] else 0,
            'n_queries': len(query_images) // len(episode['classes']) if episode['classes'] else 0,
            'classes': episode['classes']
        }
        
        if self.return_metadata:
            episode_data['support_metadata'] = support_metadata
            episode_data['query_metadata'] = query_metadata
        
        return episode_data
    
    def _process_sample(self, sample):
        """Process a single sample"""
        if len(sample) == 3:
            image, keypoints, metadata = sample
        else:
            # Fallback for different formats
            image = sample[0]
            keypoints = sample[1] if len(sample) > 1 else torch.zeros(17, 2)
            metadata = sample[2] if len(sample) > 2 else {}
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, keypoints, metadata


class EpisodicDataLoader:
    """
    DataLoader for few-shot episodic training
    Handles batching of episodes
    """
    
    def __init__(
        self,
        dataset: FewShotDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        
        # Create the underlying data loader
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, episodes):
        """Custom collate function for episodes"""
        # Stack episodes in a batch
        batched_episodes = {}
        
        for key in episodes[0].keys():
            if key in ['support_metadata', 'query_metadata', 'classes']:
                # These are lists, keep as is
                batched_episodes[key] = [ep[key] for ep in episodes]
            else:
                # These are tensors, stack them
                tensors = [ep[key] for ep in episodes]
                if all(t.numel() > 0 for t in tensors):
                    batched_episodes[key] = torch.stack(tensors)
                else:
                    batched_episodes[key] = tensors
        
        return batched_episodes
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


class NWayKShotEpisodeGenerator:
    """
    Advanced episode generator with various sampling strategies
    """
    
    def __init__(
        self,
        class_to_samples: Dict,
        n_way: int = 5,
        k_shot: int = 1,
        n_queries: int = 1,
        sampling_strategy: str = 'uniform',
        class_balanced: bool = True,
        difficulty_progressive: bool = False
    ):
        self.class_to_samples = class_to_samples
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_queries = n_queries
        self.sampling_strategy = sampling_strategy
        self.class_balanced = class_balanced
        self.difficulty_progressive = difficulty_progressive
        
        self.classes = list(class_to_samples.keys())
        
        # Pre-compute class statistics
        self._compute_class_stats()
    
    def _compute_class_stats(self):
        """Compute statistics for each class"""
        self.class_stats = {}
        
        for cls in self.classes:
            samples = self.class_to_samples[cls]
            self.class_stats[cls] = {
                'num_samples': len(samples),
                'avg_difficulty': self._estimate_difficulty(samples),
                'class_frequency': len(samples) / sum(len(s) for s in self.class_to_samples.values())
            }
    
    def _estimate_difficulty(self, samples) -> float:
        """Estimate difficulty of samples (simplified)"""
        # This could be based on sample complexity, diversity, etc.
        # For now, return random difficulty
        return random.random()
    
    def sample_classes(self) -> List:
        """Sample classes for an episode"""
        if self.sampling_strategy == 'uniform':
            return random.sample(self.classes, self.n_way)
        
        elif self.sampling_strategy == 'weighted':
            # Sample classes based on frequency
            weights = [self.class_stats[cls]['class_frequency'] for cls in self.classes]
            return random.choices(self.classes, weights=weights, k=self.n_way)
        
        elif self.sampling_strategy == 'balanced':
            # Ensure balanced representation
            available_classes = [cls for cls in self.classes 
                               if self.class_stats[cls]['num_samples'] >= self.k_shot + self.n_queries]
            return random.sample(available_classes, min(self.n_way, len(available_classes)))
        
        elif self.sampling_strategy == 'difficult':
            # Sample more difficult classes
            difficulties = [(cls, self.class_stats[cls]['avg_difficulty']) for cls in self.classes]
            difficulties.sort(key=lambda x: x[1], reverse=True)
            return [cls for cls, _ in difficulties[:self.n_way]]
        
        else:
            return random.sample(self.classes, self.n_way)
    
    def sample_episode(self) -> Dict:
        """Generate a complete episode"""
        # Sample classes
        episode_classes = self.sample_classes()
        
        # Sample support and query for each class
        episode = {
            'support': {},
            'query': {},
            'classes': episode_classes,
            'metadata': {
                'sampling_strategy': self.sampling_strategy,
                'difficulty': self._compute_episode_difficulty(episode_classes)
            }
        }
        
        for cls in episode_classes:
            samples = self.class_to_samples[cls]
            
            # Ensure we have enough samples
            required = self.k_shot + self.n_queries
            if len(samples) < required:
                print(f"Warning: Class {cls} has only {len(samples)} samples, need {required}")
                # Repeat samples if necessary
                samples = (samples * ((required // len(samples)) + 1))[:required]
            
            # Sample support and query
            sampled = random.sample(samples, required)
            episode['support'][cls] = sampled[:self.k_shot]
            episode['query'][cls] = sampled[self.k_shot:self.k_shot + self.n_queries]
        
        return episode
    
    def _compute_episode_difficulty(self, classes: List) -> float:
        """Compute overall difficulty of episode"""
        difficulties = [self.class_stats[cls]['avg_difficulty'] for cls in classes]
        return sum(difficulties) / len(difficulties)
    
    def generate_episodes(self, num_episodes: int) -> List[Dict]:
        """Generate multiple episodes"""
        episodes = []
        for _ in range(num_episodes):
            episodes.append(self.sample_episode())
        return episodes


class FewShotBatchSampler:
    """
    Custom batch sampler for few-shot learning
    """
    
    def __init__(
        self,
        dataset: FewShotDataset,
        n_episodes: int,
        batch_size: int = 4
    ):
        self.dataset = dataset
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        
        # Generate episode indices
        self.episode_indices = list(range(n_episodes))
    
    def __iter__(self):
        if self.batch_size > 1:
            # Batch episodes together
            for i in range(0, len(self.episode_indices), self.batch_size):
                batch = self.episode_indices[i:i + self.batch_size]
                yield batch
        else:
            # Single episode per batch
            for idx in self.episode_indices:
                yield [idx]
    
    def __len__(self):
        return math.ceil(self.n_episodes / self.batch_size)
