import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


choices = ['2', '9', '9~', '<p:>', '?', '@', 'A', 'E', 'H', 'J', 'O', 'R', 'S',
           'Z', 'a', 'a~', 'b', 'd', 'e', 'e~', 'f', 'g', 'i', 'j', 'k', 'l',
           'm', 'n', 'o', 'o~', 'p', 's', 't', 'u', 'v', 'w', 'y', 'z', 'N']


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


def train_and_plot_cnn(train_loader, test_loader, model,
                       epochs=20, lr=10**(-4)/2):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(epochs)):
        print("epoch: ", epoch+1)
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in tqdm(train_loader, leave=False):
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

        print("testing...")
        with torch.no_grad():
            for data in tqdm(test_loader):
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


if True:
    print("loading data")
    ((train_tensors, train_phonemes),
     (_, _),
     (test_tensors, test_phonemes)) = torch.load('resized0.pth')
    # ((train_tensors2, train_phonemes2),
    #  (_, _),
    #  (_, _)) = torch.load('resized1.pth')

    # train_tensors = torch.cat([train_tensors, train_tensors2], dim=0)
    # train_phonemes = torch.cat([train_phonemes, train_phonemes2], dim=0)

    print(train_phonemes.size())
    print(train_tensors.size())
    print(train_tensors[0])
    input()

    print("data preparation")
    train_dataset = CustomDataset(train_tensors.float(),
                                  train_phonemes.float())
    test_dataset = CustomDataset(test_tensors.float(),
                                 test_phonemes.float())

    print("batching")
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    print("Creating model")
    model = CNN_Phoneme()

    print("training")
    train_and_plot_cnn(train_loader, test_loader, model, 4)

    print("saving model")
    torch.save(model.state_dict(), "model_resized.pt")
elif False:
    bags = torch.load("data_phoneme_bag20;0.1;1000_150d.pth")
    models = []
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 2

    test_data = CustomDataset(bags[2][0][:, :, :].float(),
                              bags[2][1][:, :].float())
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    for i, (X_train, Y_train) in enumerate(bags[0]):
        print("training model: ", i)
        # Initialiser un nouveau modèle pour chaque bag
        model = CNN_Phoneme()
        optimizer = torch.optim.AdamW(model.parameters(), lr=10**(-4))

        train_data = CustomDataset(X_train[:, :, :].float(),
                                   Y_train[:, :].float())
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        train_and_plot_cnn(train_loader, test_loader, model, epochs)

        models.append(model)

    print("joint accuracy: ", bagging_accuracy(models, test_loader))

    torch.save(models, "modèles50.pth")
elif False:
    bags = torch.load("data_phoneme_bag_150d.pth")
    models = []
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 5

    test_data = CustomDataset(bags[2][0][:, :, :].float(),
                              bags[2][1][:, :].float())
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    models = torch.load("modèles.pth")

    print(bagging_accuracy(models, test_loader))
elif False:
    print("loading data")
    ((train_tensors, train_phonemes),
     # (valid_tensors, valid_phonemes),
     (test_tensors, test_phonemes)) = torch.load('same.pth')

    print("data preparation")
    train_dataset = CustomDataset(train_tensors.float(),
                                  train_phonemes.float())
    test_dataset = CustomDataset(test_tensors.float(),
                                 test_phonemes.float())

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = AttentionRNN(50, 100, 100, 39)
    model.load_state_dict(torch.load("modelLLM.pt"))
    print("computing acc per classes")
    acc_per_class(model, train_loader)

exit()
print("loading data")
((train_tensors, train_phonemes),
    (test_tensors, test_phonemes)) = torch.load('same.pth')

print("data preparation")
train_dataset = CustomDataset(train_tensors[:, :, :].float(),
                              train_phonemes[:, :].float())
test_dataset = CustomDataset(test_tensors[:, :, :].float(),
                             test_phonemes[:, :].float())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = AttentionRNN(50, 100, 100, 39)

print("training")
train_and_plot_cnn(train_loader, test_loader, model, 500)

print("saving model")
torch.save(model.state_dict(), "modelLLM.pt")
