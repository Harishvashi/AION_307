
#Standard Library modules
import importlib
import operator
import platform
import time
import sys
import json
import logging

#Third Party modules
import joblib
import pandas as pd 
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

model_name = 'XGBClassifier_svd'

                    
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

                    
def scoring_criteria(score_param, problem_type, class_count):                    
    if problem_type == 'classification':                    
        scorer_mapping = {                    
                    'recall':{'binary_class': 'recall', 'multi_class': 'recall_weighted'},                    
                    'precision':{'binary_class': 'precision', 'multi_class': 'precision_weighted'},                    
                    'f1_score':{'binary_class': 'f1', 'multi_class': 'f1_weighted'},                    
                    'roc_auc':{'binary_class': 'roc_auc', 'multi_class': 'roc_auc_ovr_weighted'}                    
                   }                    
        if (score_param.lower() == 'roc_auc') and (class_count > 2):                    
            score_param = make_scorer(sklearn.metrics.roc_auc_score, needs_proba=True,multi_class='ovr',average='weighted')                    
        else:                    
            class_type = 'binary_class' if class_count == 2 else 'multi_class'                    
            if score_param in scorer_mapping.keys():                    
                score_param = scorer_mapping[score_param][class_type]                    
            else:                    
                score_param = 'accuracy'                    
    return score_param
                    
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

def train(base_config):
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
    prev_step_output = deployment_dict['featureengineering']['Status']                
    dataLoc = prev_step_output['DataFilePath']                
    if not Path(dataLoc).exists():                
        return {'Status':'Falure','Message':'Data location does not exists.'}                
    config_file = Path(__file__).parent/'train.json'                
    if not Path(config_file).exists():                
        return {'Status':'Falure','Message':'Config file is missing'}                
    config = read_json(config_file)                
    log_file = Path(__file__).parent.stem + '.log'                
    logging.basicConfig(filename=output_model_dir/log_file, filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%b-%y %H:%M:%S')                
    status = dict()                
    df = pd.read_csv(dataLoc)
    selected_features = prev_step_output['selected_features']
    target_feature = config['target_feature']
    train_features = prev_step_output['total_features'].copy()
    train_features.remove(target_feature)
    logging.info('Data balancing done')
    scorer = scoring_criteria(config['scoring_criteria'],config['problem_type'], df[target_feature].nunique())
    logging.info('Scoring criterio: recall')
    search_space = config['search_space']
    X_train, X_test, y_train, y_test = train_test_split(df[train_features],df[target_feature],train_size=config['train_ratio'])
    logging.info('Training XGBClassifier for svd')
            
    tried_model = []            
    features = selected_features['svd']            
    for current_model in search_space:            
        estimator = list(current_model['algo'].keys())[0]            
        import_from = list(current_model['algo'].values())[0]            
        module = importlib.import_module(import_from)            
        estimator = getattr(module, estimator)()            
        param = current_model['param']
        grid = RandomizedSearchCV(estimator, param, scoring=scorer, n_iter=config['optimization_param']['iterations'],cv=config['optimization_param']['trainTestCVSplit'])            
        grid.fit(X_train[features], y_train)            
        tried_model.append(            
            {            
                'estimator': grid.best_estimator_,            
                'best score': grid.best_score_,            
                'best params': grid.best_params_            
            }            
    )
    result = sorted(tried_model, key=operator.itemgetter('best score'),reverse=True)            
    estimator = result[0]['estimator']            
    train_score = result[0]['best score']            
    all_estimators = [ (str(x['estimator']),x['best score']) for x in tried_model]
    y_pred = estimator.predict(X_test[features])
    test_score = round(recall_score(y_test,y_pred),2) * 100
    logging.info('Confusion Matrix:\n')
    logging.info(pd.DataFrame(confusion_matrix(y_test,y_pred)))
    model_path = str(output_model_dir/(model_name + '.pkl'))                
    joblib.dump(estimator, model_path)                
    status = {'Status':'Success','ModelPath':model_path,'FeaturesUsed':features,'test_score':test_score,'train_score':train_score,'all_models':all_estimators}                
    logging.info(f'Test score {test_score}')                
    logging.info(f'Estimator {all_estimators[0]}')                
    logging.info(f'Trained model saved at {model_path}')                
    deploy_train_file = output_model_dir/(model_name + '.deploy')                
    deployment_train_dict = {}                
    deployment_train_dict['training'] = {}                
    deployment_train_dict['training'][model_name] = status                
    write_json(deployment_train_dict, deploy_train_file)                
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
        print(train(config))                
    except Exception as e:                
        logging.error(e, exc_info=True)                
        status = {'Status':'Failure','Message':str(e)}                
        print(json.dumps(status))