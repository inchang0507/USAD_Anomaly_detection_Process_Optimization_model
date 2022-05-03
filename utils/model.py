import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

    
device = get_default_device()

class Encoder(nn.Module):
    def __init__(self, in_size, latent_size):
        super().__init__()
        self.linear1 = nn.Linear(in_size, int(in_size/2))
        self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
        self.linear3 = nn.Linear(int(in_size/4), latent_size)
        self.relu = nn.ReLU(True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        
    def forward(self, w):
        out = self.linear1(w)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        z = self.relu(out)
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_size, out_size):
        super().__init__()
        self.linear1 = nn.Linear(latent_size, int(out_size/4))
        self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
        self.linear3 = nn.Linear(int(out_size/2), out_size)
        self.relu = nn.ReLU(True)
        self.leakyrelu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        out = self.linear1(z)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        w = self.sigmoid(out)
        return w
    
class UsadModel(nn.Module):
    def __init__(self, w_size, z_size):
        super().__init__()
        self.encoder = Encoder(w_size, z_size)
        self.decoder1 = Decoder(z_size, w_size)
        self.decoder2 = Decoder(z_size, w_size)
  
    def training_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
        return loss1,loss2

    def validation_step(self, batch, n):
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
        return {'val_loss1': loss1, 'val_loss2': loss2}
        
    def validation_epoch_end(self, outputs):
        batch_losses1 = [x['val_loss1'] for x in outputs]
        epoch_loss1 = torch.stack(batch_losses1).mean()
        batch_losses2 = [x['val_loss2'] for x in outputs]
        epoch_loss2 = torch.stack(batch_losses2).mean()
        return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    
def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch,device), n) for [batch] in val_loader]
    return model.validation_epoch_end(outputs)

def training(epochs, model, train_loader, val_loader, opt_func=torch.optim.AdamW):
    history = []
    optimizer1 = opt_func(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = opt_func(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    for epoch in range(epochs):
        for [batch] in train_loader:
            batch= to_device(batch,device)
            
            #Train AE1
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(batch,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
            
        result = evaluate(model, val_loader, epoch+1)
        model.epoch_end(epoch, result)
        history.append(result)
    return history
    
def testing(model,test_loader,scaler, alpha=.4, beta=.6): # 0.5/0.5 -> AUC 0.944
    col_list = ['label','Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 'Clamp_Close_Time', 'Cushion_Position', 'Plasticizing_Position','Clamp_Open_Position',
                                                            'Max_Injection_Speed', 'Max_Screw_RPM', 'Average_Screw_RPM', 'Max_Injection_Pressure','Max_Switch_Over_Pressure', 'Max_Back_Pressure','Average_Back_Pressure', 'Barrel_Temperature_1',
                                                            'Barrel_Temperature_2','Barrel_Temperature_3', 'Barrel_Temperature_4', 'Barrel_Temperature_5','Barrel_Temperature_6', 'Hopper_Temperature', 'Mold_Temperature_3','Mold_Temperature_4']


    std_dict = {'Average_Back_Pressure': 3.5308202674245766,
 'Average_Screw_RPM': 126.83697241477364,
 'Barrel_Temperature_1': 2.3023663231248475,
 'Barrel_Temperature_2': 1.8873562768444156,
 'Barrel_Temperature_3': 1.8543292105107274,
 'Barrel_Temperature_4': 2.0180813228406684,
 'Barrel_Temperature_5': 1.1345663583953662,
 'Barrel_Temperature_6': 0.42876223331833163,
 'Clamp_Close_Time': 0.07584310277907853,
 'Clamp_Open_Position': 42.17057721372091,
 'Cushion_Position': 0.11500382943119065,
 'Cycle_Time': 0.3722673809476143,
 'Filling_Time': 0.14028605925356127,
 'Hopper_Temperature': 2.4337818595998892,
 'Injection_Time': 0.18052585093963686,
 'Max_Back_Pressure': 1.7680847389472414,
 'Max_Injection_Pressure': 1.985763560540526,
 'Max_Injection_Speed': 1.005339814575919,
 'Max_Screw_RPM': 0.14114760040247093,
 'Max_Switch_Over_Pressure': 0.7547364101435092,
 'Mold_Temperature_3': 1.171389067338036,
 'Mold_Temperature_4': 1.3707801863197815,
 'Plasticizing_Position': 0.6483065586499827,
 'Plasticizing_Time': 0.2889456688264329,
 'label': 0.0}

    results=[]
    outputs = []
    time = 0
    for [batch] in test_loader:
        batch=to_device(batch,device)
        w1=model.decoder1(model.encoder(batch))
        w2=model.decoder2(model.encoder(w1))
        anolmaly_score = alpha*torch.mean((batch-w1)**2,axis=1)+beta*torch.mean((batch-w2)**2,axis=1)
        if anolmaly_score>= 0.04:
            print("################################# 이상 감지 #################################")
            print()
            print('time = %d'%time)
            batch_list = batch.tolist()[0]
            w1_list = w1.tolist()[0]
            batch_list.insert(0,1)
            w1_list.insert(0,1)
            real_and_w1 = [batch_list,w1_list]
            scaling_df = pd.DataFrame(real_and_w1,columns=col_list)

            inverse = scaler.inverse_transform(real_and_w1)
            inverse_df = pd.DataFrame(inverse,columns=col_list)
            ano_vari_list = []
            for i,j in zip(std_dict.keys(),std_dict.values()):
                
                
                if abs(inverse_df.loc[0,i] - inverse_df.loc[1,i]) > j:
                    print(i+'가 정상 범위보다',inverse_df.loc[0,i] - inverse_df.loc[1,i],'만큼 벗어났습니다.')
                    ano_vari_list.append(i)
            
            scaling_df = scaling_df[ano_vari_list]
            scaling_df_T = scaling_df.T.reset_index()
            plt.figure(figsize=(25,10))
            plt.scatter(scaling_df_T['index'],scaling_df_T[0],c="red",label='real(abnormal state)',s=100)
            plt.scatter(scaling_df_T['index'],scaling_df_T[1],c="green",label='fake(normal state)',s=100)
            plt.xlabel("variables")
            plt.ylabel("scaling_score")
            plt.title("time : %d"%time)
            plt.legend()
            plt.show()


            print()
            
        results.append(anolmaly_score)
        output = {}
        output['real'] = batch
        output['w1'] = w1
        output['w2'] = w2
        outputs.append(output)
        time +=1
    return results, outputs
