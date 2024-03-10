import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import numpy as np
from sklearn.linear_model import LinearRegression


choices = ['2', '9', '9~', '<p:>', '?', '@', 'A', 'E', 'H', 'J', 'O', 'R', 'S',
           'Z', 'a', 'a~', 'b', 'd', 'e', 'e~', 'f', 'g', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'o~', 'p', 's', 't', 'u', 'v', 'w', 'y', 'z', 'N']


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                               stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # self.nlayer1 = nn.LayerNorm([172, 100])  # plan
        self.nlayer1 = nn.LayerNorm([16, 100])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=1, padding=1)
        self.relu2 = nn.ReLU()
        # self.nlayer2 = nn.LayerNorm([32, 86, 50])  # plan
        self.nlayer2 = nn.LayerNorm([32, 8, 50])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(p=0.5)

        # self.fc = nn.Linear(40800, 39)
        # self.fc = nn.Linear(34400, 39)
        self.fc = nn.Linear(3200, 39)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.nlayer1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.nlayer2(x)
        x = self.pool2(x)
        x = self.dropout(x)
        # x = x.view(-1, 40800)  # mag
        # x = x.view(-1, 34400)  # gradio
        x = x.view(-1, 3200)
        x = self.fc(x)
        return x


class CNN_PhonemeOLD(nn.Module):
    def __init__(self, num_classes=39):
        super(CNN_PhonemeOLD, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(64 * (306//4) * (100//4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class CNN_Phoneme(nn.Module):
    def __init__(self, num_classes=39):
        super(CNN_Phoneme, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.1)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((38, 38))

        self.fc1 = nn.Linear(128 * 38 * 38, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.pool(self.conv3(x)))

        x = self.adaptive_pool(x)

        x = x.view(x.size(0), -1)
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


class RNNModel(nn.Module):
    def __init__(self, input_size=100, hidden_size=30, num_layers=2,
                 output_size=39):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out


class GRUModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=32, num_layers=1,
                 output_size=39):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)

        out, _ = self.gru(x)

        out = self.fc(out[:, -1, :])
        return out


def train_and_plot_cnn(train_loader, test_loader, valid_loader, model,
                       epochs=20, lr=5*10**(-4),
                       weight_decay=0.01):
    criterion = My_loss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=weight_decay)

    train_losses = []
    valid_losses = []
    test_losses = []
    train_accuracies = []
    valid_accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(epochs)):
        print("epoch: ", epoch+1)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(train_loader, leave=False):
            inputs, labels = data
            inputs = inputs.to("mps")
            labels = labels.to("mps")
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
        valid_loss = 0.0
        test_loss = 0.0
        correct = 0
        valid_correct = 0
        total = 0
        total_valid = 0

        print("testing...")
        with torch.no_grad():
            for data in tqdm(test_loader):
                images, labels = data
                images = images.to("mps")
                labels = labels.to("mps")
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                label_index = torch.argmax(labels, dim=1)
                total += labels.size(0)
                correct += (predicted == label_index).sum().item()
            for data in tqdm(valid_loader):
                images, labels = data
                images = images.to("mps")
                labels = labels.to("mps")
                outputs = model(images)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                label_index = torch.argmax(labels, dim=1)
                total_valid += labels.size(0)
                valid_correct += (predicted == label_index).sum().item()

        test_losses.append(test_loss / len(test_loader))
        valid_losses.append(valid_loss / len(valid_loader))
        test_accuracies.append(100 * correct / total)
        valid_accuracies.append(100 * valid_correct / total_valid)
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, '
              f'Train Accuracy: {train_accuracies[-1]:.2f}%, '
              f'Valid Loss: {valid_losses[-1]:.4f}, '
              f'Valid Accuracy: {valid_accuracies[-1]:.2f}%'
              f'Test Loss: {test_losses[-1]:.4f}, '
              f'Test Accuracy: {test_accuracies[-1]:.2f}%')

    # Tracé des courbes de perte et de précision
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.plot(valid_accuracies, label='Valid Accuracy')
    plt.title('Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    file = "images/" + str(random.random()) + '.png'
    plt.savefig(file)
    plt.close()


class CustomDataset(Dataset):
    def __init__(self, tensors, phonemes):
        self.tensors = tensors
        self.phonemes = phonemes

    def __len__(self):
        return len(self.phonemes)

    def __getitem__(self, idx):
        return self.tensors[idx], self.phonemes[idx]


def bagging_accuracy(models, test_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            outs = torch.stack([model(inputs) for model in models])
            pred = torch.sum(outs, dim=0)

            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            label_index = torch.argmax(labels, dim=1)
            correct += (predicted == label_index).sum().item()

    accuracy = correct / total
    return accuracy


def acc_per_class(model, test_loader):
    num_classes = len(choices)
    correct_preds = torch.zeros(num_classes)
    total_preds = torch.zeros(num_classes)

    # Passer le modèle en mode d'évaluation
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            # Obtenir les prédictions du modèle
            outputs = model(inputs)

            # Obtenir les classes prédites
            _, predicted = torch.max(outputs, 1)
            label_index = torch.argmax(labels, dim=1)
            # Mettre à jour les compteurs pour chaque classe
            for label, prediction in zip(label_index, predicted):
                if label == prediction:
                    correct_preds[label] += 1
                total_preds[label] += 1

    # Calculer l'accuracy par classe
    accuracy_per_class = correct_preds / total_preds

    # Afficher l'accuracy par classe
    for i in range(num_classes):
        print(f'Accuracy de la classe {choices[i]}: '
              f'{accuracy_per_class[i].item():.4f} '
              f' freq :' + str(100 * total_preds[i].item() /
                               len(test_loader.dataset))[:4] + '%')


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


class AttentionModule(nn.Module):
    def __init__(self, embed_size, heads):
        super(AttentionModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_size,
                                               num_heads=heads,
                                               batch_first=True)

    def forward(self, x):
        attention_output, _ = self.attention(x, x, x)
        return attention_output


class AttentionRNN(nn.Module):
    def __init__(self, input_dim, attention_dim, rnn_hidden_size, num_classes):
        super(AttentionRNN, self).__init__()

        self.attention = AttentionModule(input_dim, 2)

        self.rnn = nn.GRU(input_size=input_dim,
                          hidden_size=rnn_hidden_size,
                          num_layers=1, batch_first=True)

        # Couche de sortie
        self.fc = nn.Linear(rnn_hidden_size, num_classes)

    def forward(self, x):

        x = self.attention(x)

        x, _ = self.rnn(x)

        x = x[:, -1, :]
        x = self.fc(x)

        return x


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


if True:
    print("loading data")
    ((train_tensors, train_phonemes),
        (valid_tensors, valid_phonemes),
        (test_tensors, test_phonemes)) = torch.load('smths.pth')
    # ((train_tensors2, train_phonemes2),
    #  (_, _),
    #  (_, _)) = torch.load('resized1.pth')

    # train_tensors = torch.cat([train_tensors, train_tensors2], dim=0)
    # train_phonemes = torch.cat([train_phonemes, train_phonemes2], dim=0)

    train_tensors = train_tensors.float()
    train_phonemes = train_phonemes.float()

    valid_tensors = valid_tensors.float()
    valid_phonemes = valid_phonemes.float()

    test_tensors = test_tensors.float()
    test_phonemes = test_phonemes.float()

    train_tensors = torch.unsqueeze(train_tensors, 1)
    valid_tensors = torch.unsqueeze(valid_tensors, 1)
    test_tensors = torch.unsqueeze(test_tensors, 1)

    class_probabilities = torch.mean(train_phonemes, dim=0)
    random_accuracy = torch.sum(class_probabilities ** 2)
    print("Random train accuracy for Bayes estimator:", random_accuracy.item())
    class_probabilities = torch.mean(test_phonemes, dim=0)
    random_accuracy = torch.sum(class_probabilities ** 2)
    print("Random test accuracy for Bayes estimator:", random_accuracy.item())

    proba_x1 = torch.mean(train_phonemes, dim=0)
    proba_x2 = torch.mean(test_phonemes, dim=0)
    dot_product = torch.dot(proba_x1, proba_x2)

    print("Bayes benchmark:", dot_product.item())

    print("data preparation")
    # w, h = train_tensors.size(1), train_tensors.size(2)
    # c = train_phonemes.size(1)

    # train_tensors = train_tensors.flatten(start_dim=1)
    # test_tensors = test_tensors.flatten(start_dim=1)

    train_dataset = CustomDataset(train_tensors,
                                  train_phonemes)
    valid_dataset = CustomDataset(valid_tensors,
                                  valid_phonemes)
    test_dataset = CustomDataset(test_tensors,
                                 test_phonemes)

    print("batching")
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print("Creating model")
    # model = nn.Sequential(nn.Linear(w*h, c)).to('mps')
    model = SimpleCNN()

    del train_tensors
    del train_phonemes

    print("training")
    train_and_plot_cnn(train_loader, test_loader, valid_loader,
                       model.to("mps"), 200)

    print("saving model")
    torch.save(model.state_dict(), "new_reduced_2conv.pt")
else:
    print("loading data")
    ((train_tensors, train_phonemes),
     (valid_tensors, valid_phonemes),
     (test_tensors, test_phonemes)) = torch.load('new_reduced2.pth')

    print("data preparation")

    train_tensors = train_tensors.float()
    train_phonemes = train_phonemes.float()

    train_tensors = torch.unsqueeze(train_tensors, 1)
    test_tensors = torch.unsqueeze(test_tensors, 1)

    test_tensors = test_tensors.float()
    test_phonemes = test_phonemes.float()

    # train_tensors = torch.unsqueeze(train_tensors, 1)
    # test_tensors = torch.unsqueeze(test_tensors, 1)

    # w, h = train_tensors.size(1), train_tensors.size(2)
    # c = train_phonemes.size(1)

    # train_tensors = train_tensors.view(train_tensors.size(0), -1)
    # test_tensors = test_tensors.view(test_tensors.size(0), -1)

    train_dataset = CustomDataset(train_tensors,
                                  train_phonemes)
    test_dataset = CustomDataset(test_tensors,
                                 test_phonemes)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # model = nn.Sequential(
    #     nn.Linear(w*h, c)
    # )
    model = SimpleCNN()

    model.load_state_dict(torch.load("new_reduced_2conv.pt"))
    print("computing acc per classes")
    probs, freq = calculate_empirical_probability(test_loader, model, 1)
    reg(probs, freq)
