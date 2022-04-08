
#Standard Library modules
import json
import platform
import time
import sys
import logging

#Third Party modules
import pandas as pd 
from pathlib import Path
from sklearn.decomposition import TruncatedSVD
import joblib

                    
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

def featureSelector(base_config):
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
    prev_step_output = deployment_dict['transformation']['Status']                
    dataLoc = prev_step_output['DataFilePath']                
    if not Path(dataLoc).exists():                
        return {'Status':'Falure','Message':'Data location does not exists.'}                
    config_file = Path(__file__).parent/'selector.json'                
    if not Path(config_file).exists():                
        return {'Status':'Falure','Message':'Config file is missing'}                
    config = read_json(config_file)                
    log_file = Path(__file__).stem + '.log'                
    logging.basicConfig(filename=output_model_dir/log_file, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')                
    status = dict()                
    df = pd.read_csv(dataLoc)
    train_features = prev_step_output.get('train_features', config['train_features'])
    target_feature = config['target_feature']
    cat_features = config['cat_features']
    total_features = []
    df = df[train_features + [target_feature]]
    log_dataframe(df)
    selected_features = {}
    deployment_dict['featureengineering']= {}
    logging.info('Running dimensionality reduction technique( svd)')
    dimension_reducer = TruncatedSVD(n_components=config['n_components'], n_iter=7, random_state=42)
    pca_array = dimension_reducer.fit_transform(df[train_features])
    pca_columns = ['pca_'+str(e) for e in list(range(pca_array.shape[1]))]
    scaledDF = pd.DataFrame(pca_array, columns=pca_columns)
    scaledDF[target_feature] = df[target_feature]
    df = scaledDF
    selected_features['svd'] = pca_columns
    total_features = df.columns.tolist()
    deployment_dict['featureengineering']['feature_reducer']= {}
    reducer_file_name = str(output_model_dir/'feature_reducer.pkl')
    joblib.dump(dimension_reducer, reducer_file_name)
    deployment_dict['featureengineering']['feature_reducer']['file']= reducer_file_name
    deployment_dict['featureengineering']['feature_reducer']['features']= train_features
    log_dataframe(df)
                
    csv_path = str(output_data_dir/(usecase+'_selector'+'.csv'))                
    write_data(df, csv_path,index=False)                
    status = {'Status':'Success','DataFilePath':csv_path,'total_features':total_features, 'selected_features':selected_features}                
    logging.info(f'Selected data saved at {csv_path}')                
    deployment_dict['featureengineering']['Status'] = status                
    write_json(deployment_dict, deploy_file)                
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
        print(featureSelector(config))                
    except Exception as e:                
        logging.error(e, exc_info=True)                
        status = {'Status':'Failure','Message':str(e)}                
        print(json.dumps(status))