#DEFINIZIONE DELLA RETE NEURALE

import torch
import torch.nn as nn 
import torch.nn.functional as F

#---TASK 1: CLASSIFICAZIONE--------------------------------------------------------------
class ClassificationCNN(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationCNN, self).__init__() #chiamata al costruttore della classe padre
        
        #creiamo gli strati di convoluzione 
        #Conv2d(mappe in ingresso, mappe in uscita, dimesione del filtro 3x3)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) #il padding mantiene la dimensione spaziale dell'immagine dopo la convoluzione
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) #ogni pooling dimezza altezza e larghezza delle immagini (64-->32) 

        self.flatten_dim = 64 * 8 * 8  #mappa in uscita dall'ultima conv * altezza dopo i tre pool * larghezza dopo i 3 pool

        self.fc1 = nn.Linear(self.flatten_dim, 128) #comprimo tutte le informazioni in 128 valori importanti
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):  #definiamo come i dati devono effettivamente muoversi all'interno della rete
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flatten_dim) #passiamo da immagini ad una lista di numeri
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x) #assegnazione di un punteggio ad ogni classe
        return x


#----TASK 2: REGRESSIONE--------------------------------------------------------------------
class RegressionCNN(nn.Module):
    def __init__(self, num_classes):
        super(RegressionCNN,self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_dim = 128 * 8 * 8
        
        self.fc1_input_dim = self.flatten_dim + num_classes  #con la regressione mi serve sapere anche la classe dell'immagine
        self.fc1 = nn.Linear(self.fc1_input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x, class_idx):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        features = x.view(-1, self.flatten_dim)
        
        class_onehot = F.one_hot(class_idx, num_classes=self.num_classes).float() #trasformiamo la lasse da singolo numero ad array "binario"
        combined = torch.cat((features, class_onehot), dim=1) #combiniamo le feauture delle immagini con le info sulla classe
        
        out = F.relu(self.fc1(combined))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out




