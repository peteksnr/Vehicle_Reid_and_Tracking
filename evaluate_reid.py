import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

from osnet import osnet_ain_x1_0  # Ensure this file is in the same folder

class ReIDModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = osnet_ain_x1_0(pretrained=True)
        self.backbone.classifier = nn.Identity()  
        self.embed = nn.Linear(512, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        embed = self.embed(feat)
        embed = F.normalize(embed, p=2, dim=1)
        logits = self.classifier(embed)
        return embed, logits


class EvalDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.jpg')]
        self.transform = transform

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        label = int(os.path.basename(path).split('_')[0]) 
        if self.transform:
            img = self.transform(img)
        return img, label, path

    def __len__(self):
        return len(self.paths)


def extract_features(model, dataloader, device):
    model.eval()
    features, labels, paths = [], [], []
    with torch.no_grad():
        for imgs, lbls, pths in tqdm(dataloader):
            imgs = imgs.to(device)
            embeds, _ = model(imgs)
            features.append(embeds.cpu())
            labels.extend(lbls)
            paths.extend(pths)
    features = torch.cat(features, dim=0)
    labels = np.array(labels)
    return features, labels, paths

def evaluate(query_feats, query_ids, gallery_feats, gallery_ids):
    query_feats = F.normalize(query_feats, p=2, dim=1)
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)

    dist_matrix = torch.cdist(query_feats, gallery_feats, p=2)
    indices = dist_matrix.argsort(dim=1)

    rank1 = 0
    all_ap = []

    for i in range(len(query_ids)):
        q_id = query_ids[i]
        order = indices[i].cpu().numpy()
        matches = (gallery_ids[order] == q_id).astype(np.int32)

        if np.sum(matches) == 0:
            continue  

        if matches[0] == 1:
            rank1 += 1

        ap = average_precision_score(matches, -dist_matrix[i][order].cpu().numpy())
        all_ap.append(ap)

    rank1_score = 100. * rank1 / len(query_ids)
    mAP_score = 100. * np.mean(all_ap)
    return rank1_score, mAP_score

def draw_border(img, color, border_width=10):
    draw = ImageDraw.Draw(img)
    for i in range(border_width):
        draw.rectangle(
            [i, i, img.width - i - 1, img.height - i - 1],
            outline=color
        )
    return img

def visualize_topk_with_color(query_feats, query_paths, gallery_feats, gallery_paths, gallery_ids, output_dir, top_k=5):
    os.makedirs(output_dir, exist_ok=True)
    query_feats = F.normalize(query_feats, p=2, dim=1)
    gallery_feats = F.normalize(gallery_feats, p=2, dim=1)
    dist_matrix = torch.cdist(query_feats, gallery_feats, p=2)

    for i in tqdm(range(len(query_paths)), desc="Saving color-coded top-k results"):
        q_path = query_paths[i]
        q_img = Image.open(q_path).convert('RGB')
        q_id = int(os.path.basename(q_path).split('_')[0])

        topk_indices = torch.argsort(dist_matrix[i])[:top_k]
        q_folder = os.path.join(output_dir, f"query_{i}")
        os.makedirs(q_folder, exist_ok=True)
        q_img.save(os.path.join(q_folder, "query.jpg"))

        for rank, idx in enumerate(topk_indices):
            g_path = gallery_paths[idx]
            g_img = Image.open(g_path).convert('RGB')
            g_id = int(os.path.basename(g_path).split('_')[0])

            color = "green" if g_id == q_id else "red"
            g_img = draw_border(g_img, color)

            g_img.save(os.path.join(q_folder, f"rank{rank+1}_{os.path.basename(g_path)}"))


def main():
    device=torch.device('mps') if torch.backends.mps.is_available() else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    query_path = "path/to/your/folder/VeRi/image_query"
    gallery_path = "path/to/your/folder/VeRi/image_test"
    model_path = "path/to/your/folder/model.pth"

    query_dataset = EvalDataset(query_path, transform)
    gallery_dataset = EvalDataset(gallery_path, transform)

    query_loader = DataLoader(query_dataset, batch_size=32, shuffle=False)
    gallery_loader = DataLoader(gallery_dataset, batch_size=32, shuffle=False)

    num_classes = 575
    model = ReIDModel(num_classes=num_classes).to(device)
    state_dict = torch.load(model_path, map_location=device)

    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier.')}

    model.load_state_dict(state_dict, strict=False)

    q_feats, q_ids, q_paths = extract_features(model, query_loader, device)
    g_feats, g_ids, g_paths = extract_features(model, gallery_loader, device)

    rank1, mAP = evaluate(q_feats, q_ids, g_feats, g_ids)

    print(f"\n🔎 Evaluation Results:\nRank-1 Accuracy: {rank1:.2f}%\nmAP: {mAP:.2f}%")

    visualize_topk_with_color(
        query_feats=q_feats,
        query_paths=q_paths,
        gallery_feats=g_feats,
        gallery_paths=g_paths,
        gallery_ids=g_ids,
        output_dir="path/to/your/folder/results",  
        top_k=5
    )
    print("📸 Top-k results saved in 'results/'")

if __name__ == "__main__":
    main()