# 必要模組載入
import pandas as pd
import torch
import os
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import warnings
import matplotlib.pyplot as plt

# 取消警告顯示
warnings.filterwarnings('ignore', category=FutureWarning)

# 定義儲存路徑
save_path =  r"C:\Users\電機系味宸漢\Desktop\HW2-1"

# 載入資料
train_data = pd.read_csv(r"C:\Users\電機系味宸漢\Desktop\HW2-1\titanic\train.csv", encoding='ISO-8859-1')
test_data = pd.read_csv(r"C:\Users\電機系味宸漢\Desktop\HW2-1\titanic\test.csv", encoding='ISO-8859-1')

# 合併資料進行前處理
total_data = pd.concat([train_data, test_data], ignore_index=True)

# 移除不必要的欄位（如 Name, Ticket, Cabin 等）
total_data = total_data.drop(columns=['Name', 'Ticket', 'Cabin'])

# 處理缺失值和標籤編碼
total_data['Age'].fillna(total_data['Age'].median(), inplace=True)
total_data['Embarked'].fillna(total_data['Embarked'].mode()[0], inplace=True)
total_data['Fare'].fillna(total_data['Fare'].median(), inplace=True)

# 將 'Sex' 和 'Embarked' 欄位進行標籤編碼
label_encoder = LabelEncoder()
total_data['Sex'] = label_encoder.fit_transform(total_data['Sex'])
total_data['Embarked'] = label_encoder.fit_transform(total_data['Embarked'])

# 分割訓練與測試數據
train_data = total_data[~total_data['Survived'].isnull()]
test_data = total_data[total_data['Survived'].isnull()].drop('Survived', axis=1)

X = train_data.drop(['Survived'], axis=1)
y = train_data['Survived']

# 將資料分為訓練集和驗證集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 正規化數據
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
test_data_scaled = scaler.transform(test_data)

# 定義 PyTorch 的 Dataset
class TitanicDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)
        else:
            return torch.tensor(self.X[idx], dtype=torch.float32)

# 建立訓練集和驗證集的 Dataset 和 DataLoader
train_dataset = TitanicDataset(X_train, y_train.values)
val_dataset = TitanicDataset(X_val, y_val.values)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 定義神經網路模型
class TitanicModel(nn.Module):
    def __init__(self):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# 初始化模型、損失函數和優化器
model = TitanicModel()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
num_epochs = 2000
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch).squeeze()
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

        # 計算訓練準確率
        train_correct += ((y_pred > 0.5) == y_batch).sum().item()

    train_acc = train_correct / len(train_loader.dataset)

    # 驗證模型
    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            val_loss += loss.item() * X_batch.size(0)

            # 計算驗證準確率
            val_correct += ((y_pred > 0.5) == y_batch).sum().item()

    val_acc = val_correct / len(val_loader.dataset)

    # 打印訓練和驗證損失、準確率
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader.dataset)}, Val Loss: {val_loss/len(val_loader.dataset)}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# 訓練結束後，計算驗證集上的混淆矩陣和最終的準確率
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for X_batch, y_batch in val_loader:
        y_pred_batch = model(X_batch).squeeze()
        y_pred.extend((y_pred_batch > 0.5).int().tolist())
        y_true.extend(y_batch.int().tolist())

# 計算並打印混淆矩陣
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 視覺化混淆矩陣（黑白配色）
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Greys)
plt.title("Confusion Matrix")
plt.show()

# 最終的訓練和驗證準確率
print(f"Final Train Accuracy: {train_acc:.4f}")
print(f"Final Validation Accuracy: {val_acc:.4f}")

# 儲存模型
torch.save(model.state_dict(), os.path.join(save_path, "titanic_model.pth"))

# 加載模型進行測試
model.eval()
test_dataset = TitanicDataset(test_data_scaled)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

predictions = []
with torch.no_grad():
    for X_batch in test_loader:
        y_pred = model(X_batch).squeeze()
        predictions.extend((y_pred > 0.5).int().tolist())

# 將預測結果儲存為 CSV 文件
output = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
output.to_csv(os.path.join(save_path, 'submission.csv'), index=False)
print("Results saved to submission.csv")
