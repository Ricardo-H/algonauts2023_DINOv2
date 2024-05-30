import os
from datetime import datetime
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

print("正在加载数据...")
root_dir = os.path.join(os.getcwd(), 'data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mse_left_list = []
mse_right_list = []
start_time = datetime.now()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class DinoFeatureExtractor(nn.Module):
    def __init__(self):
        super(DinoFeatureExtractor, self).__init__()
        self.dino = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vits14')
        self.dino.head = nn.Identity()

    def forward(self, x):
        return self.dino(x)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def extract_features(loader, desc=""):
    features, left_brain_targets, right_brain_targets = [], [], []
    with torch.no_grad():
        for images, left_brain, right_brain in tqdm(loader, desc=desc):
            images = images.to(device)
            feature = dino_model(images).cpu()
            features.append(feature)
            left_brain_targets.append(left_brain)
            right_brain_targets.append(right_brain)
    return torch.cat(features), torch.cat(left_brain_targets), torch.cat(right_brain_targets)


class BrainImageDataset(Dataset):
    def __init__(self, img_dir, left_brain_data, right_brain_data):
        self.img_dir = img_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.left_brain_data = left_brain_data
        self.right_brain_data = right_brain_data

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        img = Image.open(img_path)
        img_trans = transform(img)
        img.close()
        left_brain = self.left_brain_data[idx]
        right_brain = self.right_brain_data[idx]
        return img_trans, left_brain, right_brain


def train_model(model, criterion, optimizer, X_train, y_train, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        inputs = X_train.to(device)
        targets = y_train.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model


dino_model = DinoFeatureExtractor().to(device)
dino_model.eval()
for subject in range(1, 9):
    print(f"开始处理第 [{subject}/9] 个受试者")
    subject_start_time = datetime.now()
    data_dir = os.path.join(root_dir, f'subj{subject:02}')
    if not os.path.exists(data_dir):
        print(f"未找到数据目录：{data_dir} 跳过...")
        continue

    train_dir = os.path.join(data_dir, 'training_split')
    test_dir = os.path.join(data_dir, 'test_split')

    left_brain_data = np.load(os.path.join(
        train_dir, 'training_fmri', 'lh_training_fmri.npy'))
    right_brain_data = np.load(os.path.join(
        train_dir, 'training_fmri', 'rh_training_fmri.npy'))
    test_left_brain_data_path = os.path.join(
        test_dir, 'test_fmri', 'lh_pred_test.npy')
    test_right_brain_data_path = os.path.join(
        test_dir, 'test_fmri', 'rh_pred_test.npy')
    if not os.path.exists(os.path.join(
            test_dir, 'test_fmri')):
        os.makedirs(os.path.join(test_dir, 'test_fmri'))
    train_img_dir = os.path.join(train_dir, 'training_images')
    test_img_dir = os.path.join(test_dir, 'test_images')

    assert len(left_brain_data) == len(right_brain_data) == len(
        os.listdir(train_img_dir)), "数据长度不匹配"
    print("数据加载完成")

    dataset = BrainImageDataset(
        train_img_dir, left_brain_data, right_brain_data)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = BrainImageDataset(
        test_img_dir, left_brain_data, right_brain_data)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32,
                            shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=8, pin_memory=True)

    X_train, y_train_left, y_train_right = extract_features(
        train_loader, desc="训练集特征提取")
    X_val, y_val_left, y_val_right = extract_features(
        val_loader, desc="验证集特征提取")

    input_dim = X_train.shape[1]
    output_dim_left = y_train_left.shape[1]
    output_dim_right = y_train_right.shape[1]

    model_left = LinearRegressionModel(input_dim, output_dim_left).to(device)
    model_right = LinearRegressionModel(input_dim, output_dim_right).to(device)

    criterion = nn.MSELoss()
    optimizer_left = torch.optim.Adam(
        model_left.parameters(), lr=0.001, weight_decay=1e-5)
    optimizer_right = torch.optim.Adam(
        model_right.parameters(), lr=0.001, weight_decay=1e-5)

    model_left = train_model(
        model_left, criterion, optimizer_left, X_train, y_train_left, num_epochs=100)
    model_right = train_model(
        model_right, criterion, optimizer_right, X_train, y_train_right, num_epochs=100)
    print("模型训练完成")

    model_left.eval()
    model_right.eval()
    with torch.no_grad():
        y_pred_left = model_left(X_val.to(device)).cpu()
        y_pred_right = model_right(X_val.to(device)).cpu()
    mse_left = mean_squared_error(y_val_left.numpy(), y_pred_left.numpy())
    mse_right = mean_squared_error(y_val_right.numpy(), y_pred_right.numpy())
    mse_left_list.append(mse_left)
    mse_right_list.append(mse_right)
    print(f"[{subject}/9] 模型评估完成")
    subject_end_time = datetime.now()
    print(f"本次训练用时：{subject_end_time - subject_start_time}")
    with torch.no_grad():
        X_test = extract_features(test_loader, desc="测试集特征提取")[0].to(device)
        test_pred_left = model_left(X_test).cpu().numpy()
        test_pred_right = model_right(X_test).cpu().numpy()
    np.save(test_left_brain_data_path, test_pred_left)
    np.save(test_right_brain_data_path, test_pred_right)

end_time = datetime.now()
print(f"总用时：{end_time - start_time}")
print(f"Left Brain average mse: {np.mean(mse_left_list)}")
print(f"Right Brain average mse: {np.mean(mse_right_list)}")
