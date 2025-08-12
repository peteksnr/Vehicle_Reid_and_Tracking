import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm
from osnet import osnet_ain_x1_0

device=torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
print("Using device:", device)

data_dir = "path/to/your/project/veri_data/train"


transform = T.Compose([
    T.Resize((256, 128)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0),
])

dataset = ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

class ReIDModel(nn.Module):
    def __init__(self, num_classes, num_colors=10, num_types=9):
        super().__init__()
        self.backbone = osnet_ain_x1_0(pretrained=True)
        self.backbone.classifier = nn.Identity()

        self.embed = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(512, num_classes)       
        self.color_classifier = nn.Linear(512, num_colors)  
        self.type_classifier = nn.Linear(512, num_types)   

    def forward(self, x):
        feat = self.backbone(x)
        embed = self.embed(feat)
        embed = F.normalize(embed, p=2, dim=1)
        logits = self.classifier(embed)
        color_logits = self.color_classifier(embed)
        type_logits = self.type_classifier(embed)
        return embed, logits, color_logits, type_logits



class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)

    def forward(self, embeddings, labels):
        dists = torch.cdist(embeddings, embeddings, p=2)
        anchors, positives, negatives = [], [], []

        for i in range(len(embeddings)):
            label = labels[i]
            mask_pos = (labels == label).float()
            mask_neg = (labels != label).float()

            if mask_pos.sum() <= 1 or mask_neg.sum() == 0:
                continue

            pos_dist = dists[i] * mask_pos
            pos_dist[i] = -1  
            hard_pos_idx = torch.argmax(pos_dist).item()

            neg_dist = dists[i] + (1 - mask_neg) * 1e6
            hard_neg_idx = torch.argmin(neg_dist).item()

            anchors.append(embeddings[i])
            positives.append(embeddings[hard_pos_idx])
            negatives.append(embeddings[hard_neg_idx])

        if not anchors:
            return torch.tensor(0.0, requires_grad=True).to(embeddings.device)
        
        return self.loss_fn(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))

class AdaptiveTripletLoss(nn.Module):
    def __init__(self, margin=0.3, hard_ratio=0.3, medium_ratio=0.4, easy_ratio=0.3):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin)
        self.hard_ratio = hard_ratio
        self.medium_ratio = medium_ratio
        self.easy_ratio = easy_ratio

    def forward(self, embeddings, labels):
        dists = torch.cdist(embeddings, embeddings, p=2)
        anchors, positives, negatives = [], [], []

        for i in range(len(embeddings)):
            label = labels[i]
            pos_mask = (labels == label).bool()
            neg_mask = (labels != label).bool()

            pos_indices = torch.where(pos_mask)[0]
            neg_indices = torch.where(neg_mask)[0]
            pos_indices = pos_indices[pos_indices != i]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            pos_dists = dists[i][pos_indices]
            neg_dists = dists[i][neg_indices]

            def sample_from_sorted(dists, indices, ratio):
                count = max(1, int(len(indices) * ratio))
                sorted_vals, sorted_idx = torch.sort(dists)
                selected = indices[sorted_idx[:count]]
                return selected[torch.randint(len(selected), (1,)).item()]

            hard_pos = sample_from_sorted(pos_dists, pos_indices, self.hard_ratio)
            medium_pos = sample_from_sorted(pos_dists, pos_indices, self.medium_ratio)
            easy_pos = sample_from_sorted(pos_dists, pos_indices, self.easy_ratio)

            hard_neg = sample_from_sorted(neg_dists, neg_indices, self.hard_ratio)
            medium_neg = sample_from_sorted(neg_dists, neg_indices, self.medium_ratio)
            easy_neg = sample_from_sorted(neg_dists, neg_indices, self.easy_ratio)

            anchors.extend([embeddings[i]] * 3)
            positives.extend([embeddings[hard_pos], embeddings[medium_pos], embeddings[easy_pos]])
            negatives.extend([embeddings[hard_neg], embeddings[medium_neg], embeddings[easy_neg]])

        if not anchors:
            return torch.tensor(0.0, requires_grad=True).to(embeddings.device)

        return self.loss_fn(torch.stack(anchors), torch.stack(positives), torch.stack(negatives))


model = ReIDModel(num_classes=len(dataset.classes)).to(device)
# uncomment belove if you want to use pretrained model
# model.load_state_dict(torch.load("model.pth", map_location=device)) 
ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
triplet_loss = AdaptiveTripletLoss(margin=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


def train():
    epochs = 5
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            color_labels = torch.randint(0, 10, labels.shape).to(device)
            type_labels = torch.randint(0, 9, labels.shape).to(device)

            embeds, logits, color_logits, type_logits = model(images)

            loss_id = ce_loss(logits, labels)
            loss_triplet = triplet_loss(embeds, labels)
            loss_color = ce_loss(color_logits, color_labels)
            loss_type = ce_loss(type_logits, type_labels)

            loss = loss_id + loss_triplet + 0.5 * (loss_color + loss_type)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss={avg_loss:.2f}, Accuracy={acc:.2f}%")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved.")


if __name__ == '__main__':
    train()
