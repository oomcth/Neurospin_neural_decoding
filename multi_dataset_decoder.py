import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
import torch.nn.functional as F
from model import ProcessingModel


choices = ['2', '9', '9~', '<p:>', '?', '@', 'A', 'E', 'H', 'J', 'O', 'R', 'S',
           'Z', 'a', 'a~', 'b', 'd', 'e', 'e~', 'f', 'g', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'o~', 'p', 's', 't', 'u', 'v', 'w', 'y', 'z', 'N']

urls = ['new_data_highf_270' + str(i) + '.pth' for i in range(10)]
urls = ['3022_NoOff0.pth']
model_name = "model_multi_270_highg"
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

batch_size = 64

# device = 'cpu'


class SimpleCNN(nn.Module):
    def __init__(self, dropout=0, batch=True):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        if batch:
            # self.bn1 = nn.LayerNorm([16, 270, 100])
            # self.bn1 = nn.LayerNorm([16, 306, 100])
            # self.bn1 = nn.LayerNorm([16, 83, 100])
            self.bn1 = nn.LayerNorm([16, 185, 100])
        else:
            self.bn1 = nn.Identity()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=dropout)

        # self.fc = nn.Linear(32 * 76 * 25, 39)
        # self.fc = nn.Linear(16_000, 39)
        self.fc = nn.Linear(36800, 39)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = self.dropout(x)
        # x = x.view(-1, 32 * 76 * 25)
        # x = x.view(-1, 16000)
        x = x.view(-1, 36800)
        x = self.fc(x)
        return x


class SimpleCNN3(nn.Module):
    def __init__(self, dropout=0, batch=True):
        super(SimpleCNN3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        if batch:
            self.bn1 = nn.LayerNorm([16, 270, 100])
        else:
            self.bn1 = nn.Identity()
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        if batch:
            self.bn2 = nn.LayerNorm([32, 135, 50])
        else:
            self.bn2 = nn.Identity()
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(64 * 33 * 12, 39)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.conv3(x)))

        x = self.dropout(x)
        x = x.view(-1, 64 * 33 * 12)
        x = self.fc(x)
        return x


class SimpleCNN4(nn.Module):
    def __init__(self, dropout=0):
        super(SimpleCNN4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=2)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(17856, 39)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))

        x = self.dropout(x)
        x = x.view(-1, 17856)
        x = self.fc(x)
        return x


class CNNModel(nn.Module):
    def __init__(self, dropout=0, autre=False):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([32, 270, 100])

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, 135, 50])

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, padding=1)
        self.ln3 = nn.LayerNorm([128, 67, 25])

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, padding=1)
        self.ln4 = nn.LayerNorm([256, 33, 12])

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=3, padding=1)
        self.ln5 = nn.LayerNorm([512, 16, 6])

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024,
                               kernel_size=3, padding=1)
        self.ln6 = nn.LayerNorm([1024, 8, 3])

        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=1024 * 8 * 3, out_features=1_000)
        self.fc2 = nn.Linear(in_features=1_000, out_features=39)

    def forward(self, x):
        x = self.pool(F.relu(self.ln1(self.conv1(x))))
        x = self.pool(F.relu(self.ln2(self.conv2(x))))
        x = self.pool(F.relu(self.ln3(self.conv3(x))))
        x = self.pool(F.relu(self.ln4(self.conv4(x))))
        x = self.pool(F.relu(self.ln5(self.conv5(x))))
        x = F.relu(self.ln6(self.conv6(x)))

        x = self.dropout(x)
        x = x.view(-1, 1024 * 8 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class My_loss(nn.Module):
    def __init__(self, num_classes=39):
        super(My_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.other = nn.MSELoss()

    def forward(self, x, y):
        return self.loss(x, y)


class CustomDataset(Dataset):
    def __init__(self, tensors, phonemes):
        self.tensors = tensors
        self.phonemes = phonemes

    def __len__(self):
        return len(self.phonemes)

    def __getitem__(self, idx):
        return self.tensors[idx], self.phonemes[idx]


def train_and_plot_cnn(urls, model,
                       epochs=20, lr=5*10**(-4),
                       weight_decay=0.01):
    criterion = My_loss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    ((_, _), (_, _),
     (test_tensor, test_phonemes)) = torch.load(urls[0])

    test_tensor = test_tensor
    test_phonemes = test_phonemes

    test_tensor = torch.unsqueeze(test_tensor, 1)
    test_dataset = CustomDataset(test_tensor, test_phonemes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    for epoch in tqdm(range(epochs), desc='epoch'):
        for i, url in tqdm(enumerate(np.random.permutation(urls)),
                           desc='dataset', total=len(urls), leave=False):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            ((train_tensors, train_phonemes),
             (_, _), (_, _)) = torch.load(url)

            train_tensors = train_tensors
            train_phonemes = train_phonemes

            train_tensors = torch.unsqueeze(train_tensors, 1)
            train_dataset = CustomDataset(train_tensors,
                                          train_phonemes)
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True)
            for data in tqdm(train_loader, leave=False, desc='batch'):
                inputs, labels = data

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                label_index = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == label_index).sum().item()

            train_losses.append(running_loss / len(train_loader))
            train_accuracies.append(100 * correct / total)

            model.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data in tqdm(test_loader, leave=False, desc='testing'):
                    images, labels = data
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    label_index = torch.argmax(labels, dim=1)
                    total += labels.size(0)
                    correct += (predicted == label_index).sum().item()

            test_losses.append(test_loss / len(test_loader))
            test_accuracies.append(100 * correct / total)

            print("")
            print(" \n ")
            print(f'Epoch {(epoch+1 + i / len(urls)):.2f}:\n'
                  f'Train Loss: {train_losses[-1]:.4f} : '
                  f'Train Accuracy: {train_accuracies[-1]:.2f}%;\n'
                  f'Test Loss: {test_losses[-1]:.4f} : '
                  f'Test Accuracy: {test_accuracies[-1]:.2f}%')
    torch.save(model.state_dict(), model_name + str(epoch) + ".pt")

    # Tracé des courbes de perte et de précision
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    torch.save(model.state_dict(), "model_multi_270_highg.pt")
    file = "images/" + str(random.random()) + '.png'
    plt.savefig(file)
    plt.close()

    print("computing acc per classes")
    probs, freq = calculate_empirical_probability(test_loader, model, 1)
    reg(probs, freq)


def calculate_empirical_probability(loader, model, n):
    model.eval()
    num_correct = [0] * 39
    num_samples = [0] * 39
    class_freq = np.zeros(39)
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        print("computing acc")
        for inputs, targets in tqdm(loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.topk(outputs, n, dim=1)
            for i in range(targets.size(0)):
                class_idx = targets[i].argmax().item()
                num_samples[class_idx] += 1
                class_freq[class_idx] += 1
                if class_idx in predicted[i]:
                    num_correct[class_idx] += 1
                    total_correct += 1
                total_samples += 1
    probs = [num_correct[i] / (num_samples[i] + 10**(-8)) for i in range(39)]
    total_prob = total_correct / total_samples
    print(f"Total accuracy for top {n} predictions: {total_prob:.4f}")
    for i in range(39):
        print(f'Class {choices[i]}: Accuracy = {probs[i]:.4f},'
              f' Frequency = {(100 * class_freq[i] / total_samples):.4f}')
    return probs, class_freq


def reg(probs, class_freq):
    probs = np.array(probs)
    class_freq = np.array(class_freq)

    class_freq = class_freq.reshape(-1, 1)

    model = LinearRegression()
    model.fit(class_freq, probs)

    r_squared = model.score(class_freq, probs)

    print("If we take into account prob=0")
    print("Coefficients: ", model.coef_)
    print("Intercept: ", model.intercept_)
    print("R-squared: ", r_squared)

    predictions = model.predict(class_freq)

    probs = np.array(probs)
    class_freq = np.array(class_freq)

    class_freq = class_freq.reshape(-1, 1)

    nonzero_mask = probs != 0
    probs_filtered = probs[nonzero_mask]
    class_freq_filtered = class_freq[nonzero_mask]

    model2 = LinearRegression()
    model2.fit(class_freq_filtered, probs_filtered)

    r_squared = model2.score(class_freq, probs)

    print("If we don't take into account prob=0")
    print("Coefficients: ", model2.coef_)
    print("Intercept: ", model2.intercept_)
    print("R-squared: ", r_squared)

    predictions2 = model2.predict(class_freq)

    total = class_freq.sum()

    plt.scatter(class_freq, probs, label='Data Points')
    plt.plot(np.linspace(0, class_freq.max(), 1000),
             np.linspace(0, class_freq.max(), 1000) / total,
             label='Bayes Line')
    plt.plot(class_freq, predictions, color='red',
             label='Regression Line')
    plt.plot(class_freq, predictions2, color='orange',
             label='Regression Line without zeros')
    plt.xlabel('Class Frequency')
    plt.ylabel('Probability')
    plt.title('Linear Regression of Probability on Class Frequency')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print(f"Utilisation du périphérique : {device}")

    n_epoch = 50
    if len(sys.argv) > 1:
        n_epoch = int(sys.argv[1])
    train_and_plot_cnn(urls, ProcessingModel().to(device),
                       n_epoch, lr=10**(-4), weight_decay=0.01)
