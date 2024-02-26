import torch.nn as nn
import torch.functional as F

# lr = 10-4 ; 17% test acc 10 epoch max avec 20_000 per listenning.
# gros overfitting sur le train, train acc et loss linéaire en
# epoch quand arrêt
# 10-5 simmillaire ; pareil pour 5.10^-4 ---- 10-4 semble à
# la loiuche etre le mieux
# bonne acc sur les phonemes les plus fréquents.
# faire de la data avec une freq de phoneme plus équilibrée ???
# le modèle tir au piff entre quelques des phonemes les plus fréquents ?


# bagging 5 40% pas ouf joint 16.6 = test individuelle
# probablement que les modeles trouvent tous la meme chose
# baggings 20 10% 17.5 le bagging augmente les perfs des
# modèles qui sont sur 15-16 de acc

class CNN_Phoneme(nn.Module):
    def __init__(self, num_classes=39):
        super(CNN_Phoneme, self).__init__()
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


# autre modèle simpliste :
# test acc = bruit blanc autour de 13 %
# train acc monte doucement mais surement
# tout les phonemes étudié (ceux assez fréquents) ont leurs acc
# qui crois a peut pres à la meme vitesse
# sauf <p:> ; s qui est très haut en train et test (0.7) là ou les
# autre sont entre 0.2 et 0.4 (@, d)
# n et m pas fou
# certain  phoneme ont des difficulté (a, u, y, e, p très mauvais en
# train et test)
# est ce que le modèle apprends la largeur des phonemes ?
# les phonemes qui ont des durées significativement plus
# distinctes que les autres ont des meilleures perds


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


# en faisant overfitt sur le test 1 il y a des trucs étrange
# par exemple bonne généralisation et a ccuracy opur <p:> et s (60-75)
# R, i très bonne train 0 en test ????
# pas mal de truc qui ont un train bon et un test à peu près 10 points en
# dessous
# fit sur la largeur ptetre tj vrai
# mauvaise généralisation sur les phonemes de longeur commune
# moins bonne acc sur le train pour ces memes phonemes
# exempel d et l perf moyenne car largeur communes. Si on somme
# leurs acc ont retrouve des acc pour des phonemes seul 
# comme s (en réalité moins nottament car @ doit interférer)
# s est vrm seul
# seul truc proche de s est A qui est très peu présent 0.12 train ??? 
# probablement aléatoire et rien à en tirer. 0 sur test
        

# conclu : il faut faire des analyse sur des largeur fixe sans rajouter
# des pixels vides.
# étirer les images est peu être pas une super bonne idée.
# idée faire un peu comme wavetovec avec des imprimeur locaux avec multiple
# target.
