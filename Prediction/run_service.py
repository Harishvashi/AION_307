
from http.server import BaseHTTPRequestHandler,HTTPServer
from socketserver import ThreadingMixIn
import os
from os.path import expanduser
import platform
import threading
import subprocess
import argparse
import re
import cgi
import json
import sys

config_input = None

class HTTPRequestHandler(BaseHTTPRequestHandler):

	def do_POST(self):
		print('PYTHON ######## REQUEST ####### STARTED')
		if None != re.search('/AION/', self.path):
			ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
			if ctype == 'application/json':
				length = int(self.headers.get('content-length'))
				data = self.rfile.read(length)
				model = self.path.split('/')[-2]
				operation = self.path.split('/')[-1]
				data = json.loads(data)
				dataStr = json.dumps(data)
				isdir = True
				if isdir:
					if operation.lower() == 'predict':
						predict_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'Prediction.py')
						outputStr = subprocess.check_output([sys.executable,predict_path,config_input,dataStr])
						outputStr = outputStr.decode('utf-8')
						outputStr = re.search(r'predictions:(.*)',str(outputStr), re.IGNORECASE).group(1)
						outputStr = outputStr.strip()
						resp = outputStr
					else:
						outputStr = json.dumps({'Status':'Error','Msg':'Operation not supported'})
				else:
					outputStr = json.dumps({'Status':'Error','Msg':'Model Not Present'})
			else:
				outputStr = json.dumps({'Status':'ERROR','Msg':'Content-Type Not Present'})
			resp = outputStr
			resp=resp+'\n'
			resp=resp.encode()
			self.send_response(200)
			self.send_header('Content-Type', 'application/json')
			self.end_headers()
			self.wfile.write(resp)
		else:
			print('python ==> else1')
			self.send_response(403)
			self.send_header('Content-Type', 'application/json')
			self.end_headers()
			print('PYTHON ######## REQUEST ####### ENDED')
		return
	def getModelFeatures(self,modelSignature):
		datajson = {'Body':'Gives the list of features'}
		home = expanduser('~')
		if platform.system() == 'Windows':
			predict_path = os.path.join(home,'AppData','Local','HCLT','AION','target',modelSignature,'featureslist.py')
		else:
			predict_path = os.path.join(home,'HCLT','AION','target',modelSignature,'featureslist.py')
		if(os.path.isfile(predict_path)):
			outputStr = subprocess.check_output([sys.executable,predict_path])
			outputStr = outputStr.decode('utf-8')
			outputStr = re.search(r'features:(.*)',str(outputStr), re.IGNORECASE).group(1)
			outputStr = outputStr.strip()
			displaymsg = outputStr
			#displaymsg = json.dumps(displaymsg)
			return(True,displaymsg)
		else:
			displaymsg = json.dumps({'status':'ERROR','msg':'Unable to fetch featuers'})
		return(False,displaymsg)

	def getFeatures(self,modelSignature):
		datajson = {'Body':'Gives the list of features'}
		urltext = '/AION/UseCase_Version/features'
		if modelSignature != '':
			status,displaymsg = self.getModelFeatures(modelSignature)
			if status:
				urltext = '/AION/'+modelSignature+'/features'
			else:
				displaymsg = json.dumps(datajson)
		else:
			displaymsg = json.dumps(datajson)
		msg='''
URL:{url}
RequestType: POST
Content-Type=application/json
Output: {displaymsg}.
			'''.format(url=urltext,displaymsg=displaymsg)
		return(msg)

	def features_help(self,modelSignature):
		home = expanduser('~')
		if platform.system() == 'Windows':
			display_path = os.path.join(home,'AppData','Local','HCLT','AION','target',modelSignature,'display.json')
		else:
			display_path = os.path.join(home,'HCLT','AION','target',modelSignature,'display.json')
		datajson = {'Body':'Data Should be in JSON Format'}
		if(os.path.isfile(display_path)):
			with open(display_path) as file:
				config = json.load(file)
			file.close()
			datajson={}
			for feature in config['numericalFeatures']:
				if feature != config['targetFeature']:
					datajson[feature] = 'Numeric Value'
			for feature in config['nonNumericFeatures']:
				if feature != config['targetFeature']:
					datajson[feature] = 'Category Value'
			for feature in config['textFeatures']:
				if feature != config['targetFeature']:
					datajson[feature] = 'Category Value'
		displaymsg = json.dumps(datajson)
		return(displaymsg)
	def predict_help(self,modelSignature):
		if modelSignature != '':
			displaymsg = self.features_help(modelSignature)
			urltext = '/AION/'+modelSignature+'/predict'
		else:
			datajson = {'Body':'Data Should be in JSON Format'}
			displaymsg = json.dumps(datajson)
			urltext = '/AION/UseCase_Version/predict'
		msg='''
URL:{url}
RequestType: POST
Content-Type=application/json
Body: {displaymsg}
Output: prediction,probability(if Applicable),remarks corresponding to each row.
			'''.format(url=urltext,displaymsg=displaymsg)
		return(msg)
	def performance_help(self,modelSignature):
		if modelSignature != '':
			urltext = '/AION/'+modelSignature+'/performance'
		else:
			urltext = '/AION/UseCase_Version/performance'
		datajson = {'trainingDataLocation':'Reference Data File Path','currentDataLocation':'Latest Data File Path'}
		displaymsg = json.dumps(datajson)
		msg='''
URL:{url}
RequestType: POST
Content-Type=application/json
Body: {displaymsg}
Output: HTML File Path.'''.format(url=urltext,displaymsg=displaymsg)
		return(msg)
	def monitoring_help(self,modelSignature):
		if modelSignature != '':
			urltext = '/AION/'+modelSignature+'/monitoring'
		else:
			urltext = '/AION/UseCase_Version/monitoring'
		datajson = {'trainingDataLocation':'Reference Data File Path','currentDataLocation':'Latest Data File Path'}
		displaymsg = json.dumps(datajson)
		msg='''
URL:{url}
RequestType: POST
Content-Type=application/json
Body: {displaymsg}
Output: Affected Columns. HTML File Path.'''.format(url=urltext,displaymsg=displaymsg)
		return(msg)
	def explain_help(self,modelSignature):
		if modelSignature != '':
			displaymsg = self.features_help(modelSignature)
			urltext = '/AION/'+modelSignature+'/explain'
		else:
			datajson = {'Body':'Data Should be in JSON Format'}
			displaymsg = json.dumps(datajson)
			urltext = '/AION/UseCase_Version/explain'
		msg='''
URL:{url}
RequestType: POST
Content-Type=application/json
Body: {displaymsg}
Output: anchor (Local Explanation),prediction,forceplot,multidecisionplot.'''.format(url=urltext,displaymsg=displaymsg)
		return(msg)
	def help_text(self,modelSignature):
		predict_help = self.predict_help(modelSignature)
		explain_help = self.explain_help(modelSignature)
		features_help = self.getFeatures(modelSignature)
		monitoring_help = self.monitoring_help(modelSignature)
		performance_help = self.performance_help(modelSignature)
		msg='''
Following URL:

Prediction
{predict_help}

Local Explaination
{explain_help}

Features
{features_help}

Monitoring
{monitoring_help}

Performance
{performance_help}
'''.format(predict_help=predict_help,explain_help=explain_help,features_help=features_help,monitoring_help=monitoring_help,performance_help=performance_help)
		return msg

	def do_GET(self):
		print('PYTHON ######## REQUEST ####### STARTED')
		if None != re.search('/AION/', self.path):
			self.send_response(200)
			self.send_header('Content-Type', 'application/json')
			self.end_headers()
			helplist = self.path.split('/')[-1]
			print(helplist)
			if helplist.lower() == 'help':
				model = self.path.split('/')[-2]
				if model.lower() == 'aion':
					model =''
				msg = self.help_text(model)
			elif helplist.lower() == 'predict':
				model = self.path.split('/')[-2]
				if model.lower() == 'aion':
					model =''
				msg = self.predict_help(model)
			elif helplist.lower() == 'explain':
				model = self.path.split('/')[-2]
				if model.lower() == 'aion':
					model =''
				msg = self.explain_help(model)
			elif helplist.lower() == 'monitoring':
				model = self.path.split('/')[-2]
				if model.lower() == 'aion':
					model =''
				msg = self.monitoring_help(model)
			elif helplist.lower() == 'performance':
				model = self.path.split('/')[-2]
				if model.lower() == 'aion':
					model =''
				msg = self.performance_help(model)
			elif helplist.lower() == 'features':
				model = self.path.split('/')[-2]
				if model.lower() == 'aion':
					model =''
				status,msg = self.getModelFeatures(model)
			else:
				model = self.path.split('/')[-2]
				if model.lower() == 'aion':
					model =helplist
				msg = self.help_text(model)
			self.wfile.write(msg.encode())
		else:
			self.send_response(403)
			self.send_header('Content-Type', 'application/json')
			self.end_headers()
		return

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
	allow_reuse_address = True

	def shutdown(self):
		self.socket.close()
		HTTPServer.shutdown(self)

class SimpleHttpServer():
	def __init__(self, ip, port):
		self.server = ThreadedHTTPServer((ip,port), HTTPRequestHandler)

	def start(self):
		self.server_thread = threading.Thread(target=self.server.serve_forever)
		self.server_thread.daemon = True
		self.server_thread.start()

	def waitForThread(self):
		self.server_thread.join()

	def stop(self):
		self.server.shutdown()
		self.waitForThread()

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='HTTP Server')
	parser.add_argument('config', help='Configuration file which contain usecase info')
	args = parser.parse_args()

	server = SimpleHttpServer('0.0.0.0', 60051)
	config_input = args.config
	print('HTTP Server Running...........')
	server.start()
	server.waitForThread()