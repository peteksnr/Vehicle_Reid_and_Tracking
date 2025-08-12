import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np


class ReIDModel(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        from osnet import osnet_x1_0  
        self.backbone = osnet_x1_0(pretrained=False)
        self.backbone.classifier = nn.Identity()
        self.embed = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        embed = self.embed(feat)
        embed = F.normalize(embed, p=2, dim=1)
        logits = self.classifier(embed)
        return embed, logits

class EmbeddingBuffer:
    def __init__(self, max_length=30):
        self.buffer = {} 
        self.max_length = max_length

    def update(self, track_id, embedding):
        if track_id not in self.buffer:
            self.buffer[track_id] = []
        self.buffer[track_id].append(embedding.detach().cpu())
        if len(self.buffer[track_id]) > self.max_length:
            self.buffer[track_id].pop(0)

    def get_average(self, track_id):
        if track_id in self.buffer and len(self.buffer[track_id]) > 0:
            return torch.stack(self.buffer[track_id]).mean(dim=0)
        return None


class FeatureExtractor:
    def __init__(self, model_path, device):
        self.model = ReIDModel()
        state_dict = torch.load(model_path, map_location=device)
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("classifier")}
        self.model.load_state_dict(filtered, strict=False)
        self.model.to(device).eval()
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def extract(self, img_crop):
        img = Image.fromarray(img_crop).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embed, _ = self.model(img)
        return embed.squeeze(0)


def cosine_distance(a, b):
    if a.device != b.device:
        b = b.to(a.device)
    return 1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def match_detections_to_tracks(detections, tracks, extractor, buffer, threshold=0.4):
    matches = []
    next_track_id = max(tracks) + 1 if tracks else 1

    for det_id, crop in detections:
        det_feat = extractor.extract(crop)
        best_match = None
        min_dist = float('inf')

        for track_id in tracks:
            past_avg = buffer.get_average(track_id)
            if past_avg is None:
                continue
            dist = cosine_distance(det_feat, past_avg)
            if dist < threshold and dist < min_dist:
                best_match = track_id
                min_dist = dist

        if best_match is not None:
            buffer.update(best_match, det_feat)
            matches.append((det_id, best_match))
        else:
            buffer.update(next_track_id, det_feat)
            matches.append((det_id, next_track_id))
            next_track_id += 1

    return matches