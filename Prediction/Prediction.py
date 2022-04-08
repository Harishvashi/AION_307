
#Standard Library modules
import sys
import math
import json
import platform

#Third Party modules
import joblib
import sklearn
import numpy as np 
import pandas as pd 
from pathlib import Path
from xgboost import XGBClassifier

                    
def read_json(file_path):                    
    data = None                    
    with open(file_path,'r') as f:                    
        data = json.load(f)                    
    return data                    
                    
def write_json(data, file_path):                    
    with open(file_path,'w') as f:                    
        json.dump(data, f)                    
                    
def read_data(file_path, encoding='utf-8', sep=','):                    
    return pd.read_csv(file_path, encoding=encoding, sep=sep)                    
                    
def write_data(data, file_path, index=False):                    
    return data.to_csv(file_path, index=index)                    
                    
#Uncomment and change below code for google storage                    
#def write_data(data, file_path, index=False):                    
#    file_name= file_path.name                    
#    data.to_csv('output_data.csv')                    
#    storage_client = storage.Client()                    
#    bucket = storage_client.bucket('aion_data')                    
#    bucket.blob('prediction/'+file_name).upload_from_filename('output_data.csv', content_type='text/csv')                    
#    return data                    
                    
def is_file_name_url(file_name):                    
    supported_urls_starts_with = ('gs://')                    
    return file_name.startswith(supported_urls_starts_with)                    


class deploy():

    def __init__(self, base_config):
        self.usecase = base_config['modelName'] + '_' + base_config['modelVersion']
        self.dataLocation = base_config['dataLocation']
        home = Path.home()
        if platform.system() == 'Windows':
            from pathlib import WindowsPath
            output_data_dir = WindowsPath(home)/'AppData'/'Local'/'HCLT'/'AION'/'Data'
            output_model_dir = WindowsPath(home)/'AppData'/'Local'/'HCLT'/'AION'/'target'/self.usecase
        else:
            from pathlib import PosixPath
            output_data_dir = PosixPath(home)/'HCLT'/'AION'/'Data'
            output_model_dir = PosixPath(home)/'HCLT'/'AION'/'target'/self.usecase
        if not output_model_dir.exists():
            raise ValueError(f'Configuration file not found at {output_model_dir}')
        deploy_file = output_model_dir/'deploy.json'
        if deploy_file.exists():
            deployment_dict = read_json(deploy_file)
        else:
            raise ValueError(f'Configuration file not found: {deploy_file}')

        self.selected_features = deployment_dict['transformation']['train_features']
        self.output_model_dir = output_model_dir
        self.model_info = self.__get_best_score_config(output_model_dir)
        self.model = joblib.load(self.model_info['ModelPath'])
        self.train_features = self.model_info['FeaturesUsed']
        self.missing_values = deployment_dict['transformation']['fillna']
        self.target_encoder = joblib.load(deployment_dict['transformation']['target_encoder'])
        self.normalizer = joblib.load(deployment_dict['transformation']['normalizer']['file'])
        self.normalizer_col = deployment_dict['transformation']['normalizer']['features']
        self.feature_reducer = joblib.load(deployment_dict['featureengineering']['feature_reducer']['file'])
        self.feature_reducer_cols = deployment_dict['featureengineering']['feature_reducer']['features']
    
    def __get_best_score_config(self, output_model_dir):                
        config_files = list(Path(output_model_dir).glob('*.deploy'))                
        if not config_files:                
            raise ValueError('Training output status file not found')
        score = -math.inf
        best_config = None                
        for file in config_files:                
            data = read_json(file)                
            if 'training' in data:                
                model_name = list(data['training'].keys())[0]
                if data['training'][model_name]['test_score'] > score:
                    score = data['training'][model_name]['test_score']                
                    best_config = data['training'][model_name]                
        return best_config

    def predict(self, data=None):
        if not data:
            data = self.dataLocation
        df = pd.DataFrame()
        if Path(data).exists():
            if Path(data).suffix == '.tsv':
                df=read_data(data,encoding='utf-8',sep='\t')
            elif Path(data).suffix == '.csv':
                df=read_data(data,encoding='utf-8')
            else:
                if Path(data).suffix == '.json':
                    jsonData = read_json(data)
                    df = pd.json_normalize(jsonData)
        elif is_file_name_url(data):
            df = read_data(data,encoding='utf-8')
        else:
            jsonData = json.loads(data)
            df = pd.json_normalize(jsonData)
        if len(df) == 0:
            raise ValueError('No data record found')
        missing_features = [x for x in self.selected_features if x not in df.columns]
        if missing_features:
            raise ValueError(f'some feature/s is/are missing: {missing_features}')
        df_copy = df.copy()
        df = df[self.selected_features]
        df.fillna(self.missing_values, inplace=True)
        df[self.normalizer_col] = self.normalizer.transform(df[self.normalizer_col])
        df = self.feature_reducer.transform(df[self.feature_reducer_cols])
        output = pd.DataFrame(self.model.predict_proba(df), columns=self.target_encoder.classes_)
        if data == self.dataLocation:
            from datetime import datetime
            date_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            df_copy['prediction'] = output.idxmax(axis=1)
            pred_loc = Path(self.dataLocation).parent/(self.usecase+'_prediction_'+date_time_str+'.csv')
            df_copy.to_csv(pred_loc,index=False)
            df_copy = pd.DataFrame({'prediction': [str(pred_loc)]})
        else:
            df_copy['prediction'] = output.idxmax(axis=1)
            df_copy['probability'] = output.max(axis=1).round(2)
            df_copy['remarks'] = output.apply(lambda x: x.to_json(), axis=1)
        output = df_copy.to_json(orient='records')
        return output

if __name__ == '__main__':
    try:
        if len(sys.argv) < 2:
            raise ValueError('config file not present')
        config = sys.argv[1]
        if Path(config).is_file() and Path(config).suffix == '.json':
            config = read_json(config)
        else:
            config = json.loads(config)
        data = None
        if len(sys.argv) > 2:
            data = sys.argv[2]
        predictor = deploy(config)
        output = predictor.predict(data)
        status = {'Status':'Success','Message':json.loads(output)}
        print('predictions:'+json.dumps(status))
    except Exception as e:
        status = {'Status':'Failure','Message':str(e)}
        print('predictions:'+json.dumps(status))