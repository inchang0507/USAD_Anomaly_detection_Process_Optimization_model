import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
import torch


class Preprocessor():
    def __init__(self, DIR, window_size=1 ,BATCH_SIZE=32, hidden_size= 12):
        self.DIR = DIR
        self.machine_name = "650톤-우진2호기"
        self.product_name = ["CN7 W/S SIDE MLD'G LH", "CN7 W/S SIDE MLD'G RH", "RG3 MOLD'G W/SHLD, LH", "RG3 MOLD'G W/SHLD, RH"]
        self.window_size = window_size
        self.BATCH_SIZE =  BATCH_SIZE
        self.hidden_size = hidden_size
        print("Preprocessor is Operating")
        
    def preprocessing_4_train(self):
        data = pd.read_csv(self.DIR)

        def make_input(data, machine_name ,product_name):
            machine_ = data['EQUIP_NAME'] == machine_name
            product_ = data['PART_NAME'] == product_name
            data = data[machine_ & product_]
            
            # 불필요하다고 판단된 columns
            data.drop(['_id','TimeStamp','PART_FACT_PLAN_DATE', 'Reason',
                       #'ERR_FACT_QTY',
                       'PART_FACT_SERIAL','PART_NAME','EQUIP_CD', 'EQUIP_NAME',
                       # mean=0인거 제거
                       'Mold_Temperature_1', 'Mold_Temperature_2', 'Mold_Temperature_5', 
                       'Mold_Temperature_6', 'Mold_Temperature_7', 'Mold_Temperature_8', 
                       'Mold_Temperature_9', 'Mold_Temperature_10', 'Mold_Temperature_11', 
                       'Mold_Temperature_12', 'Barrel_Temperature_7', 'Switch_Over_Position'], 
                      axis=1, inplace=True)
            return data

        
        
        # "650톤-우진2호기'의 "CN7 W/S SIDE MLD'G LH" 데이터만 변수를 제거하여 가져옴
        cn7lh = make_input(data, self.machine_name, self.product_name[0])
        # "650톤-우진2호기'의 "CN7 W/S SIDE MLD'G RH" 데이터만 변수를 제거하여 가져옴
        cn7rh = make_input(data, self.machine_name, self.product_name[1])
        ## 동일한 제품의 LH와 RH는 합쳐줌
        cn7 = pd.concat([cn7lh, cn7rh], ignore_index=True)


        cn7['PassOrFail'] = cn7['PassOrFail'].replace('Y', 0).replace('N', 1)
        cn7.loc[cn7['Average_Screw_RPM']>=200,'Average_Screw_RPM'] = cn7.loc[cn7['Average_Screw_RPM']>=200,'Average_Screw_RPM']*0.1

        
        scaler = MinMaxScaler()
        col_name = cn7.columns
        s_train = scaler.fit_transform(cn7)
        cn7 = pd.DataFrame(s_train,columns=col_name)

        
        # 양품
        cn7_Y = cn7[cn7['PassOrFail']==0]
        # 불량
        cn7_N = cn7[cn7['PassOrFail']==1]


        X_train = cn7_Y[:5000]
        X_test = cn7_Y[5000:]
        train_df = X_train.copy()
        test_df =  pd.concat([X_test,cn7_N],axis=0)
        labels = [ float(label!= 0 ) for label  in test_df["PassOrFail"].values]
        train_df.drop(['PassOrFail'] ,axis=1, inplace=True)
        test_df.drop(['PassOrFail'] ,axis=1, inplace=True)


        windows_normal=train_df.values[np.arange(self.window_size)[None, :] + np.arange(train_df.shape[0]-self.window_size)[:, None]]
        windows_attack=test_df.values[np.arange(self.window_size)[None, :] + np.arange(test_df.shape[0]-self.window_size)[:, None]]



        w_size=windows_normal.shape[1]*windows_normal.shape[2]
        z_size=windows_normal.shape[1]* self.hidden_size

        windows_normal_train = windows_normal[:int(np.floor(.85 *  windows_normal.shape[0]))]
        windows_normal_val = windows_normal[int(np.floor(.85 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

        train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],w_size]))
        ) , batch_size=self.BATCH_SIZE, shuffle=False, num_workers=0)

        val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],w_size]))
        ) , batch_size=self.BATCH_SIZE, shuffle=False, num_workers=0)

        test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
            torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],w_size]))
        ) , batch_size=1, shuffle=False, num_workers=0)


        col_names = train_df.columns

        return train_loader, val_loader, test_loader, scaler, labels, w_size, z_size, self.window_size, col_names, cn7_Y
