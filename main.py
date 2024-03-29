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


choices = ['2', '9', '9~', '<p:>', '?', '@', 'A', 'E', 'H', 'J', 'O', 'R', 'S',
           'Z', 'a', 'a~', 'b', 'd', 'e', 'e~', 'f', 'g', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'o~', 'p', 's', 't', 'u', 'v', 'w', 'y', 'z', 'N']

urls = ['new_data_highf_270' + str(i) + '.pth' for i in range(10)]
model_name = "model_multi_270_highg"


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(32 * 67 * 25, 39)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.dropout(x)
        x = x.view(-1, 32 * 67 * 25)
        x = self.fc(x)
        return x


class SimpleCNN3(nn.Module):
    def __init__(self):
        super(SimpleCNN3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(64 * 33 * 12, 39)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.dropout(x)
        x = x.view(-1, 64 * 33 * 12)
        x = self.fc(x)
        return x


class SimpleCNN4(nn.Module):
    def __init__(self):
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

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Linear(64 * 13 * 5, 39)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.dropout(x)
        x = x.view(-1, 64 * 13 * 5)
        x = self.fc(x)
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
    test_tensor = torch.unsqueeze(test_tensor, 1)
    test_dataset = CustomDataset(test_tensor, test_phonemes)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    for epoch in tqdm(range(epochs), desc='epoch'):
        for url in tqdm(urls, desc='dataset'):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            ((train_tensors, train_phonemes),
             (_, _), (_, _)) = torch.load(url)
            # train_tensors = train_tensors[:10]
            # train_phonemes = train_phonemes[:10]
            train_tensors = torch.unsqueeze(train_tensors, 1)
            train_dataset = CustomDataset(train_tensors,
                                          train_phonemes)
            train_loader = DataLoader(train_dataset, batch_size=64,
                                      shuffle=True)
            for data in tqdm(train_loader, leave=False, desc='batch'):
                inputs, labels = data
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
                for data in tqdm(test_loader, desc='testing'):
                    images, labels = data
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    label_index = torch.argmax(labels, dim=1)
                    total += labels.size(0)
                    correct += (predicted == label_index).sum().item()

            test_losses.append(test_loss / len(test_loader))
            test_accuracies.append(100 * correct / total)

            print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, '
                  f'Train Accuracy: {train_accuracies[-1]:.2f}%, '
                  f'Test Loss: {test_losses[-1]:.4f}, '
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
    n_epoch = 2
    if len(sys.argv) > 1:
        n_epoch = int(sys.argv[1])
    train_and_plot_cnn(urls, SimpleCNN(), n_epoch, lr=10**(-4))
