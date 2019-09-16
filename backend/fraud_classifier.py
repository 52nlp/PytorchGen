# Author: Yu-Hsuan Chen (Albert)
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.autograd import Variable
from datetime import datetime
import torch.utils.data as data
import visdom
import matplotlib.pyplot as plt

data_set = "fraud"
data_dic = {"fraud_orig": "./original_data/credicard_normalized_order.pkl"}
# for training, if we have cuda, than just use cuda!
cuda = True if torch.cuda.is_available() else False
print('cuda status', cuda)
use_cuda = False  # for inference test_set, we do not use cuda.


class Visualizations:
    def __init__(self, env_name=None):
        if env_name is None:
            env_name = str(datetime.now().strftime("%d-%m %Hh%M"))
        self.env_name = env_name
        self.vis = visdom.Visdom(env=self.env_name)
        self.loss_win = None

    def plot_loss(self, loss, step):
        self.loss_win = self.vis.line(
            [loss],
            [step],
            win=self.loss_win,
            update='append' if self.loss_win else None,
            opts=dict(
                xlabel='Step',
                ylabel='Loss',
                title='Loss (mean per 10 steps)',
            )
        )


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc1_norm = torch.nn.BatchNorm1d(self.hidden_size)
        self.fc1_drop = torch.nn.Dropout(0.4)
        self.relu = torch.nn.LeakyReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        hidden_norm = self.fc1_norm(hidden)
        hidden_drop = self.fc1_drop(hidden_norm)
        relu = self.relu(hidden_drop)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output


def train_classifier(augment_file="./fake_data/fraud/wgan_2019-09-08_30000.csv", n_samples=20000, num_epochs=5):
    print("augment_file:", augment_file)
    print("default_n_samples:", n_samples)
    # read files
    df_orig = pd.read_pickle(data_dic['fraud_orig'])
    df_fake = pd.read_csv(augment_file)
    # to use the number of samples you want
    n_samples = min(df_fake.shape[0], n_samples)
    print ("fake data shape:", df_fake.shape[0])
    print ('n_samples:',n_samples)
    df_fake = df_fake.sample(n=n_samples, random_state=12)

    y = df_orig['Class']
    data_cols = list(df_orig.columns[df_orig.columns != 'Class'])
    X = df_orig[data_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    y_fake = df_fake['Class']
    df_fake.drop(['Class'], axis=1, inplace=True)
    X_train = pd.concat([X_train, df_fake])
    y_train = pd.concat([y_train, y_fake])

    # do shuffle
    X_train.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    shuffle_index = np.random.permutation(X_train.index)
    X_train = X_train.reindex(shuffle_index)
    y_train = y_train.reindex(shuffle_index)

    training_input = torch.FloatTensor(X_train.values)
    training_output = torch.FloatTensor(y_train.values.reshape(-1, 1))
    test_input = torch.FloatTensor(X_test.values)
    test_output = torch.FloatTensor(y_test.values.reshape(-1, 1))

    train_tensor = data.TensorDataset(training_input, training_output)
    test_tensor = data.TensorDataset(test_input, test_output)
    trainloader = data.DataLoader(train_tensor, batch_size=64, shuffle=True)
    testloader = data.DataLoader(test_tensor, batch_size=64, shuffle=True)
    dataloaders = {'train': trainloader, 'val': testloader}
    dataset_sizes = {'train': len(trainloader.dataset), 'val': len(testloader.dataset)}

    input_size = training_input.size()[1]  # number of features selected
    hidden_size = 18  # number of nodes/neurons in the hidden layer
    model = Net(input_size, hidden_size)  # create the model
    criterion = torch.nn.BCELoss()  # works for binary classification
    # without momentum parameter
    # optimizer = torch.optim.SGD(model.parameters(), lr = 0.9) #with momentum parameter
    optimizer = torch.optim.Adam(model.parameters())

    epochs = num_epochs
    errors = []
    #vis = Visualizations()
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                # print('train...')
            else:
                model.eval()  # Set model to evaluate mode
                # print('val...')
            running_loss = 0.0
            running_corrects = 0

            for step, (input, output) in enumerate(dataloaders[phase]):
                # zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass
                y_pred = model(input)  # Compute Loss
                loss = criterion(y_pred, output)
                if phase == 'train':
                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    errors.append(loss.item())
                    # print('Epoch {}: train loss: {}'.format(epoch, loss.item()))  # Backward pass

                    if step % 10 == 0:
                        # print(phase)
                        #vis.plot_loss(np.mean(errors), step)
                        errors.clear()
                elif phase == 'val':
                    pass
                running_loss += loss.item()

                preds = (y_pred > 0.5)
                running_corrects += torch.sum(preds == output.byte()).item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # print('size:', dataset_sizes[phase])

    return model, testloader


def eval_test_set(model, testloader):
    net = model
    net.eval()

    output_list = []
    target_list = []
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        output_list.extend(outputs.flatten().detach().numpy().tolist())
        target_list.extend(targets.flatten().detach().numpy().tolist())

    t = target_list
    o = output_list
    validation_predictions = list([1 if item > 0.5 else 0 for item in o])

    labels = t

    cm = confusion_matrix(labels, validation_predictions)

    acc = accuracy_score(labels, validation_predictions)
    recall = recall_score(labels, validation_predictions)
    precision = precision_score(labels, validation_predictions)
    f1 = f1_score(labels, validation_predictions)
    print('acc:', acc)
    print('recall:', recall)
    print('precision:', precision)
    print('f1:', f1)
    print('cm:', cm)

    return acc, recall, precision, f1, cm


#aug_file = "./fake_data/fraud/wgan_2019-09-08_30000.csv"
# aug_file = "./fake_data/fraud/smote_2019-09-08_100_5_30000.csv"

#aug_model, testloader = train_classifier(augment_file=aug_file, n_samples=20000)

#model_acc, model_recall, model_precision, model_f1, model_cm = eval_test_set(aug_model, testloader)
