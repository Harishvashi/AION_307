
#Standard Library modules
import platform
import time
import json
import sys
import logging

#Third Party modules
import pandas as pd 
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

                    
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

                    
def log_dataframe(df, msg=None):                    
    import io                    
    buffer = io.StringIO()                    
    df.info(buf=buffer)                    
    if msg:                    
        log_text = f'Data frame after {msg}:'                    
    else:                    
        log_text = 'Data frame:'                    
    log_text += '\n\t'+str(df.head(2)).replace('\n','\n\t')                    
    log_text += ('\n\t' + buffer.getvalue().replace('\n','\n\t'))                    
    logging.info(log_text)

def transformation(base_config):
                
    usecase = base_config['modelName'] + '_' + base_config['modelVersion']                
    home = Path.home()                
    if platform.system() == 'Windows':                
        from pathlib import WindowsPath                
        output_data_dir = WindowsPath(home)/'AppData'/'Local'/'HCLT'/'AION'/'Data'                
        output_model_dir = WindowsPath(home)/'AppData'/'Local'/'HCLT'/'AION'/'target'/usecase                
    else:                
        from pathlib import PosixPath                
        output_data_dir = PosixPath(home)/'HCLT'/'AION'/'Data'                
        output_model_dir = PosixPath(home)/'HCLT'/'AION'/'target'/usecase                
    if not output_model_dir.exists():                
        raise ValueError(f'Configuration file not found at {output_model_dir}')                
    deploy_file = output_model_dir/'deploy.json'                
    if deploy_file.exists():                
        deployment_dict = read_json(deploy_file)                
    else:                
        raise ValueError(f'Configuration file not found: {deploy_file}')                
    dataLoc = deployment_dict['load_data']['status']['DataFilePath']                
    if not Path(dataLoc).exists():                
        return {'Status':'Falure','Message':'Data location does not exists.'}                
    config_file = Path(__file__).parent/'transformer.json'                
    if not Path(config_file).exists():                
        return {'Status':'Falure','Message':'Config file is missing'}                
    config = read_json(config_file)                
    status = dict()                
                
    target_feature = config['target_feature']                
    train_features = [x for x in config['train_features'] if x != target_feature]                
    num_features = [x for x in config['num_features'] if x != target_feature]                
    log_file = Path(__file__).stem + '.log'                
    logging.basicConfig(filename=output_model_dir/log_file, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')                
    df = pd.read_csv(dataLoc)                
    log_dataframe(df)                
    df = df[[target_feature] + train_features]                
    deployment_dict['transformation'] = {}				
    deployment_dict['transformation']['train_features'] = train_features                
    df = df.dropna(axis=0, subset=[target_feature])                
    df = df.dropna(axis=0, how='all', subset=df.columns)                
    df = df.drop_duplicates(keep='first')                
    df = df.reset_index(drop=True)
    target_encoder = LabelEncoder()
    df[config['target_feature']] = target_encoder.fit_transform(df[config['target_feature']])
    target_encoder_file_name = str(output_model_dir/'target_encoder.pkl')
    joblib.dump(target_encoder, target_encoder_file_name)
    deployment_dict['transformation']['target_encoder'] = target_encoder_file_name
    logging.info('Categorical to numeric conversion done for target feature')
    deployment_dict['transformation']['fillna'] = {}
    cat_features = [x for x in config['cat_features'] if x != target_feature]
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_features + config['cat_num_features']] = cat_imputer.fit_transform(df[cat_features + config['cat_num_features']])
    for index, col in enumerate(cat_features + config['cat_num_features']):
        deployment_dict['transformation']['fillna'][col] = cat_imputer.statistics_[index]
    logging.info(f'Missing values replaced for categorical columns {cat_features}')
    num_imputer = KNNImputer(n_neighbors=2, weights='uniform')
    df[num_features] = num_imputer.fit_transform(df[num_features])
    for index, col in enumerate(num_features):
        deployment_dict['transformation']['fillna'][col] = num_imputer.statistics_[index]
    logging.info(f'Missing values replaced for numeric columns {num_features}')
    normalizer = MinMaxScaler()
    df[num_features] = normalizer.fit_transform(df[num_features])
    normalizer_file_name = str(output_model_dir/'normalizer.pkl')
    joblib.dump(normalizer, normalizer_file_name)
    deployment_dict['transformation']['normalizer'] = {}
    deployment_dict['transformation']['normalizer']['file'] = normalizer_file_name
    deployment_dict['transformation']['normalizer']['features'] = num_features
    logging.info(f'Normalization done for numeric columns {num_features}')                
    log_dataframe(df)                
    csv_path = str(output_data_dir/(usecase+'_transformation'+'.csv'))                
    write_data(df, csv_path,index=False)                
    status = {'Status':'Success','DataFilePath':csv_path, 'text_profiler':{}}                
    deployment_dict['transformation']['Status'] = status                
    write_json(deployment_dict, deploy_file)                
    logging.info(f'Transformed data saved at {csv_path}')                
    logging.info(f'output: {status}')                
    return json.dumps(status)
                
if __name__ == '__main__':                
    try:                
        if len(sys.argv) < 2:                
            raise ValueError('config file not present')                
        config = sys.argv[1]                
        if Path(config).is_file() and Path(config).suffix == '.json':                
            config = read_json(config)                
        else:                
            config = json.loads(config)                
        print(transformation(config))                
    except Exception as e:                
        logging.error(e, exc_info=True)                
        status = {'Status':'Failure','Message':str(e)}                
        print(json.dumps(status))