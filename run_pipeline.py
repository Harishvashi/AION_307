
from pathlib import Path
import subprocess
import sys
import json

if len(sys.argv) != 2:
    print("usage: python run_pipeline.py config_file\nPlease provide config file")
    exit()

config_file = sys.argv[1]
cwd = Path(__file__).parent

load_file = str(cwd/'DataIngestion'/'load_data.py')
transformer_file = str(cwd/'DataTransformation'/'transformer.py')
selector_file = str(cwd/'FeatureEngineering'/'selector.py')
train_folder = cwd/'ModelTraining'
deploy_file = str(cwd/'Prediction'/'Prediction.py')

cmd = ['python', load_file,config_file]
with open(config_file, 'r') as f:
    data = json.load(f)
    if not data['dataLocation']:
        print("Error: config file is corrupt.")
        exit()
result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)    
result = json.loads(result)
if result['Status'] == 'Failure':
    exit()
cmd = ['python', transformer_file,config_file]

result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)
result = json.loads(result)
if result['Status'] == 'Failure':
    exit()

cmd = ['python', selector_file,config_file]

result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)
result = json.loads(result)
if result['Status'] == 'Failure':
    exit()

train_models = [f for f in train_folder.iterdir() if f.is_dir()]
for model in train_models:
    cmd = ['python', str(model/'train.py'),config_file]
    train_result = subprocess.check_output(cmd)
    train_result = train_result.decode('utf-8')
    print(train_result)    
cmd = ['python', deploy_file,config_file]

result = subprocess.check_output(cmd)
result = result.decode('utf-8')
print(result)
