import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AdamW
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import os
import shutil


class HF_dataset(torch.utils.data.Dataset):
    """
    dataset class to pass data to llm, 
    it also contains non-sequentail features and labels
    """
    def __init__(self, input_ids, attention_masks, non_seq,labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.non_sequentials = non_seq

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(self.input_ids[index]),
            "attention_mask": torch.tensor(self.attention_masks[index]),
            "non_sequential": torch.tensor(self.non_sequentials[index]),
            "labels": torch.tensor(self.labels[index]),
        }

class Classifier_non_binary(nn.Module):
    """
    non_binary classifier with dropout regularization
    """
    def __init__(self, hidden_state_size, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_state_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(hidden_dim+24, hidden_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x, y):
        tmp = self.relu(self.fc1(self.dropout(x)))#lowering the dimension of embedding
        return self.out((self.relu(self.fc2(torch.cat([tmp, y],axis=1)))))
    
class Classifier_binary(nn.Module):
    def __init__(self, hidden_state_size, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_state_size, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(hidden_dim+24, hidden_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x, y):
        tmp = self.relu(self.fc1(self.dropout(x)))
        return self.out((self.relu(self.fc2(torch.cat([tmp, y],axis=1)))))

def load_dataframe(train_path, val_path, test_path):
    
    #load train, val, tst dataframes
    df_train = pd.read_feather(train_path)
    df_val = pd.read_feather(val_path)
    df_test = pd.read_feather(test_path)
    return df_train, df_val, df_test


def add_cell_spec_label(df):
    #finds which system dhs belongs to
    df['Label'] = np.where(df.iloc[:,-15:].to_numpy()==1)[1]+1
    return df

def add_cancer_label(df):
    #finds if the cell is cancer or not
    df['Label'] = np.where(df.iloc[:,-2:].to_numpy()==1)[1]+1
    return df


def prepare_seq(df):
    """
    creates a list of sequences and labels
    there is the option to change the code fast 
    for llms using k-mer
    """
    kmers = []
    labels = []
    for seq, label in zip(df['sequence'], df["Label"]):
        kmers.append(seq)
        labels.append(label-1)
    return kmers, labels

def prepare_non_seq(df, scaler, mask_first_part=0.0):
    """
    normalizes continous non_sequentail feature, concat it with binary features.
    for the first task we mask the continous non-sequential features. 
    """
    p1 = scaler.transform(df.iloc[:,2:10].to_numpy()) * mask_first_part#(start, end width,...)
    p2 = df.iloc[:,11:27]#components of DHS
    non_seq = torch.from_numpy(np.concatenate([p1, p2], axis=1))
    return non_seq

def create_bert_dataset(tokenizer, seq, non_seq, labels, seq_max_len=512):
    #apply tokenizer to data to pass it to LLM
    encoding = tokenizer.batch_encode_plus(
    seq,
    max_length=seq_max_len,
    padding=True,  # pad to max len
    truncation=True,  # truncate to max len
    return_attention_mask=True,
    return_tensors="pt",  # return pytorch tensors
    )
    dataset = HF_dataset(
        encoding["input_ids"], encoding["attention_mask"], non_seq, labels
    )
    return dataset

def get_embedding(model, dataloader, batch_size = 128, one_hot=True):
    #run dataset through DNABERT-2 once to prevent runing in future for training speed up.
    embedding = None
    non_seq = None
    label = None
    for i in dataloader:#run in batches to prevent system collapse
        tmp = torch.mean(model(i['input_ids'])[0], dim=1)
        tmp = tmp.detach()
        if embedding is None:
            embedding = tmp
            non_seq = i['non_sequential']
            label = i['labels']
        else:
            embedding = torch.cat([embedding, tmp], axis=0)
            non_seq = torch.cat([non_seq, i['non_sequential']],axis=0)
            label = torch.cat([label, i['labels']], axis=0)

        print(embedding.shape)
        print(non_seq.shape)
        print(label.shape)
    if one_hot:#for task one we create one hot label as target
        label_onehot = torch.nn.functional.one_hot(label, num_classes=15)
        dataset = TensorDataset(embedding, non_seq, label, label_onehot)
    else:
        dataset = TensorDataset(embedding, non_seq, label, label)
    return DataLoader(dataset, batch_size=batch_size)

def train_task1(classifier, train_loader, val_loader, optimizer, num_epochs, device, result_path):
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    log_file_path = os.path.join(result_path, 'log.txt')
    model_path = os.path.join(result_path, 'best_non_binary.pt')
    log_file = open(log_file_path, 'a')
    for epoch in range(num_epochs):
        epoch_loss_train = 0.0
        classifier.train()
        for i in train_loader:
            embedding = i[0].to(device)#obtained embedding from LLM
            non_seq = i[1].to(device).type(torch.float32)
            label = i[2].to(device).type(torch.float32)#vector of labels
            target = i[3].to(device).type(torch.float32)#one hot coded labels
            optimizer.zero_grad()
            logit = classifier(embedding, non_seq)
            loss = criterion(logit,target)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.detach().cpu().item()
        epoch_loss_train = epoch_loss_train/len(train_loader)
        if epoch%10==0:  
            if epoch%500==0:
                print(epoch)
            
            with torch.no_grad():
                s = 0
                classifier.eval()
                for i in val_loader:
                    embedding = i[0].to(device)
                    non_seq = i[1].to(device).type(torch.float32)
                    label = i[2].to(device).type(torch.float32)
                    target = i[3].to(device).type(torch.float32)
                    logit = classifier(embedding, non_seq)
                    pred = torch.argmax(logit, dim=1)
                    s = s+ (pred == label).sum()
                    loss = criterion(logit,target)
                    #print(loss)
                    
                acc = (s.cpu()/2000.0).numpy()
                epoch_log = "Epoch: " + str(epoch) + "\tLoss: " + str(epoch_loss_train) + "\tValidation Accuracy: " + str(acc) +"\n"
                log_file.writelines(epoch_log)
                #print('Epoch: ', epoch, 'Loss: ', epoch_loss_train, "Validation Accuracy: ", acc)
            
                if acc > best_val_acc:
                    best_val_acc = acc
                    
                    torch.save(classifier.state_dict(), model_path)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    log_file.close()
    return classifier

def train_task2(classifier, train_loader, val_loader, optimizer, num_epochs, device, result_path):
    criterion = nn.BCEWithLogitsLoss()
    best_val_acc = 0.0
    log_file_path = os.path.join(result_path, 'log.txt')
    model_path = os.path.join(result_path, 'best_binary.pt')
    log_file = open(log_file_path, 'a')
    
    for epoch in range(num_epochs):
        epoch_loss_train = 0.0
        classifier.train()
        for i in train_loader:
            embedding = i[0].to(device)
            non_seq = i[1].to(device).type(torch.float32)
            label = i[2].to(device).type(torch.float32)
            weight = (1/label.mean())**label#inverse weighting loss to deal with imbalanced dataset
            target = i[3].to(device).type(torch.float32)
            optimizer.zero_grad()
            logit = classifier(embedding, non_seq).squeeze()
            loss = criterion(logit,target)
            loss = torch.mean(loss * weight)
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.detach().cpu().item()
        epoch_loss_train = epoch_loss_train/len(train_loader)

        if epoch%10==0:  
            if epoch%500 == 0:
                print(epoch)
            
            with torch.no_grad():
                s = 0
                classifier.eval()
                with torch.no_grad():
                    for i in val_loader:
                        embedding = i[0].to(device)
                        non_seq = i[1].to(device).type(torch.float32)
                        label = i[2].to(device).type(torch.float32)
                        target = i[3].to(device).type(torch.float32)
                        logit = classifier(embedding, non_seq).squeeze()
                        pred = torch.round(nn.Sigmoid()(logit))
                        s = s+ (pred == label).sum()
                acc = (s.cpu()/2000.0).numpy()#assumed test datasize is 2000, needs to change with different test size. TODO 
                epoch_log = "Epoch: " + str(epoch) + "\tLoss: " + str(epoch_loss_train) + "\tValidation Accuracy: " + str(acc) +"\n"
                log_file.writelines(epoch_log)
                
                if acc > best_val_acc:
                    best_val_acc = acc
                    
                    torch.save(classifier.state_dict(), model_path)
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    return classifier

def test_task2(classifier, test_loader, device, result_path):
    model_path = os.path.join(result_path, 'best_binary.pt')
    log_file_path = os.path.join(result_path, 'log.txt')
    log_file = open(log_file_path, 'a')
    
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    with torch.no_grad():
        s = 0
        classifier.eval()
        with torch.no_grad():
            for i in test_loader:
                embedding = i[0].to(device)
                non_seq = i[1].to(device).type(torch.float32)
                label = i[2].to(device).type(torch.float32)
                target = i[3].to(device).type(torch.float32)
                logit = classifier(embedding, non_seq).squeeze()
                pred = torch.round(nn.Sigmoid()(logit))
                s = s+ (pred == label).sum()
        #print(s/2000, loss, epoch)

        
        
        acc = (s.cpu()/2000.0).numpy()
        precision = precision_score(label.cpu().numpy(), pred.cpu().numpy())
        recall = recall_score(label.cpu().numpy(), pred.cpu().numpy())
        test_results = 'Test accuracy: '+str(acc)+ '\tPrecision: '+ str(precision)+ '\tRecall'+str(recall)
        #with open(log_file_path) as f:
        log_file.writelines("test performance:\n")
        log_file.writelines(test_results)
        log_file.close()
        print('Test accuracy: ', acc, 'Precision: ', precision, 'Recall',recall)

        #print('Epoch: ', acc, recall, precision)
            



def test_task1(classifier, test_loader, device,result_path):
    model_path = os.path.join(result_path, 'best_non_binary.pt')
    log_file_path = os.path.join(result_path, 'log.txt')
    log_file = open(log_file_path, 'a')
    
    classifier.load_state_dict(torch.load(model_path))
    classifier.eval()
    with torch.no_grad():
        s = 0
        classifier.eval()
        for i in test_loader:
            embedding = i[0].to(device)
            non_seq = i[1].to(device).type(torch.float32)
            label = i[2].to(device).type(torch.float32)
            target = i[3].to(device).type(torch.float32)
            logit = classifier(embedding, non_seq)
            pred = torch.argmax(logit, dim=1)
            s = s+ (pred == label).sum()
            #loss = criterion(logit,target)
            #print(loss)
            
        acc = (s.cpu()/2000.0).numpy()
        
        precision = precision_score(label.cpu().numpy(), pred.cpu().numpy(), average='weighted')
        recall = recall_score(label.cpu().numpy(), pred.cpu().numpy(), average='weighted')
        cm = confusion_matrix(label.cpu().numpy(), pred.cpu().numpy())
        df_cm = pd.DataFrame(cm, index=[i for i in ['Hematopoietic', 'Hepatic', 'Genitourinary', 'Digestive',       'Nervous', 'Epithelial', 'Integumentary', 'Connective',
       'Cardiovascular', 'Embryonic', 'Respiratory', 'Renal',
       'Musculoskeletal', 'Endocrine', 'Fetal Life Support']],columns=[i for i in ['Hematopoietic', 'Hepatic', 'Genitourinary', 'Digestive',       'Nervous', 'Epithelial', 'Integumentary', 'Connective',
       'Cardiovascular', 'Embryonic', 'Respiratory', 'Renal',
       'Musculoskeletal', 'Endocrine', 'Fetal Life Support']])
        plt.figure(figsize=(15,13))
        sn.heatmap(df_cm, annot=True, cmap='crest')
        plt.savefig(os.path.join(result_path, "CM.png"))
        test_results = 'Test accuracy: '+str(acc)+ '\tPrecision: '+ str(precision)+ '\tRecall'+str(recall)
        #with open(log_file_path) as f:
        log_file.writelines("test performance:\n")
        log_file.writelines(test_results)
        log_file.close()
        print('Test accuracy: ', acc, 'Precision: ', precision, 'Recall',recall)


def run_task1(args):
    torch.manual_seed(0)
    np.random.seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(args.results_folder):
        shutil.rmtree(args.results_folder)
    os.mkdir(args.results_folder)
    log_file = os.path.join(args.results_folder, "log.txt")
    with open(log_file, 'w') as lf:
        lf.writelines(device)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    
    train_path = os.path.join(args.data_dir, "cell_specifity_train.ftr")
    val_path = os.path.join(args.data_dir, "cell_specifity_val.ftr")
    test_path = os.path.join(args.data_dir, "cell_specifity_test.ftr")
    #load data
    tr, val, tst = load_dataframe(train_path, val_path, test_path)
    #add label
    tr = add_cell_spec_label(tr)
    val = add_cell_spec_label(val)
    tst = add_cell_spec_label(tst)
    #prepare sequences 
    seq_train, labels_train = prepare_seq(tr)
    seq_val, labels_val = prepare_seq(val)
    seq_tst, labels_tst = prepare_seq(tst)
    #scale non_seq features 
    scaler = StandardScaler().fit(tr.iloc[:,2:10].to_numpy())
    non_seq_tr = prepare_non_seq(tr, scaler, 0.0)
    non_seq_val = prepare_non_seq(val, scaler, 0.0)
    non_seq_tst = prepare_non_seq(tst, scaler, 0.0)
    #get LLM embedding
    train_dataset = create_bert_dataset(tokenizer, seq_train, non_seq_tr, labels_train)
    train_dl = DataLoader(train_dataset, batch_size=args.batch_size)
    train_dl = get_embedding(model, train_dl)

    val_dataset = create_bert_dataset(tokenizer, seq_val, non_seq_val, labels_val)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size)
    val_dl = get_embedding(model, val_dl)

    tst_dataset = create_bert_dataset(tokenizer, seq_tst, non_seq_tst, labels_tst)
    tst_dl = DataLoader(tst_dataset, batch_size=128)
    tst_dl = get_embedding(model, tst_dl, batch_size=2000)#one batch to simplify performance evaluation
    

    clf = Classifier_non_binary(768, 32, 15).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=0.000)
    clf = train_task1(clf, train_dl, val_dl, optimizer, args.epochs, device=device, result_path=args.results_folder)
    test_task1(clf, tst_dl, device=device, result_path=args.results_folder)






def run_task2(args):
    torch.manual_seed(0)
    np.random.seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if os.path.exists(args.results_folder):
        shutil.rmtree(args.results_folder)
    os.mkdir(args.results_folder)
    log_file = os.path.join(args.results_folder, "log.txt")
    with open(log_file, 'w') as lf:
        lf.writelines(device)
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    train_path = os.path.join(args.data_dir, "cancer_train.ftr")
    val_path = os.path.join(args.data_dir, "cancer_val.ftr")
    test_path = os.path.join(args.data_dir, "cancer_test.ftr")

    tr, val, tst = load_dataframe(train_path, val_path, test_path)

    tr = add_cancer_label(tr)
    val = add_cancer_label(val)
    tst = add_cancer_label(tst)
    
    seq_train, labels_train = prepare_seq(tr)
    seq_val, labels_val = prepare_seq(val)
    seq_tst, labels_tst = prepare_seq(tst)
    
    scaler = StandardScaler().fit(tr.iloc[:,2:10].to_numpy())#column 2 to 10 contains continous non_seq features(like start end width ....)
    non_seq_tr = prepare_non_seq(tr, scaler, mask_first_part=1.0)
    non_seq_val = prepare_non_seq(val, scaler, mask_first_part=1.0)
    non_seq_tst = prepare_non_seq(tst, scaler, mask_first_part =1.0)

    train_dataset = create_bert_dataset(tokenizer, seq_train, non_seq_tr, labels_train)
    train_dl = DataLoader(train_dataset, batch_size=128)
    train_dl = get_embedding(model, train_dl, one_hot=False)

    val_dataset = create_bert_dataset(tokenizer, seq_val, non_seq_val, labels_val)
    val_dl = DataLoader(val_dataset, batch_size=128)
    val_dl = get_embedding(model, val_dl, one_hot=False)

    tst_dataset = create_bert_dataset(tokenizer, seq_tst, non_seq_tst, labels_tst)
    tst_dl = DataLoader(tst_dataset, batch_size=128)
    tst_dl = get_embedding(model, tst_dl, batch_size=2000, one_hot=False)
    

    clf = Classifier_binary(768, 64, 1).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=args.lr, weight_decay=0.000)
    clf = train_task2(clf, train_dl, val_dl, optimizer, args.epochs, device=device, result_path=args.results_folder)
    test_task2(clf, tst_dl, device=device, result_path=args.results_folder)


    



