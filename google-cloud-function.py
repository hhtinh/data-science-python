from google.cloud import storage

STG_BUCKET = 'vcm-data-playground-vcm'
STG_FOLDER = 'tmp-json/'

client = storage.Client()

def upload(fileName, xml_data):
    stg_bucket = client.get_bucket(STG_BUCKET)
    stg_blob = stg_bucket.blob(STG_FOLDER + 'Copy_' + fileName)
    stg_blob.upload_from_string(xml_data)
    print(f'Uploaded {fileName} to "{STG_BUCKET}" bucket.')

def process(data, context):
    bucketName = data['bucket']
    filePath = data['name']    
    bucket = client.get_bucket(bucketName)
    xml_blob = bucket.get_blob(filePath)
    xml_data = xml_blob.download_as_string()    
    upload(filePath.split('/')[-1], xml_data)
	
	



import xml.etree.ElementTree as ET
import xmltodict
import json
import pandas as pd
from collections import OrderedDict
from google.cloud import storage

STG_BUCKET = 'vcm-data-playground-vcm'
STG_FOLDER = 'tmp-json/'

client = storage.Client()

tables = ['_-POSDW_-E1BPCREDITCARD', \
'_-POSDW_-E1BPLINEITEMDISCOUNT', \
'_-POSDW_-E1BPLINEITEMTAX', \
'_-POSDW_-E1BPREFUNDDETAILS', \
'_-POSDW_-E1BPRETAILLINEITEM', \
'_-POSDW_-E1BPTENDER', \
'_-POSDW_-E1BPTENDEREXTENSIONS', \
'_-POSDW_-E1BPTRANSACTEXTENSIO', \
'_-POSDW_-E1BPTRANSACTION', \
'_-POSDW_-E1BPTRANSACTIONLOYAL', \
'_-POSDW_-E1BPTAXTOTALS', \
'_-POSDW_-E1BPLINEITEMVOID', \
]

def upload(fileName, stg_data):
    stg_bucket = client.get_bucket(STG_BUCKET)
    stg_blob = stg_bucket.blob(STG_FOLDER + fileName)
    stg_blob.upload_from_string(stg_data)
    print(f'Uploaded {fileName} to "{STG_BUCKET}" bucket.')

def process(data, context):
    bucketName = data['bucket']
    filePath = data['name']    
    bucket = client.get_bucket(bucketName)
    xml_blob = bucket.get_blob(filePath)
    xml_str = xml_blob.download_as_string()
    
    for table in tables:
        xmlroot = ET.fromstring(xml_str)
        xmlstr = ET.tostring(xmlroot, encoding='utf8', method='xml')
        xmldata = xmltodict.parse(xmlstr)
        try:
            xmldoc = xmldata["_-POSDW_-POSTR_CREATEMULTIPLE04"]["IDOC"]["_-POSDW_-E1POSTR_CREATEMULTIP"][table]
        except:            
            pass # print('No data for ' + table)
        else:
            if type(xmldoc) is OrderedDict:
                tmp = []
                tmp.append(xmldoc)
                xmldoc = tmp
            try:
                df = pd.DataFrame(xmldoc)
                df['SourceFile'] = filePath                 
            except:                
                print(filePath + ' - Error converting to DataFrame for ' + table)                
                print(str(type(xmldoc)))
            else:
                upload((filePath.split('/')[-1]).split('.')[0] + table.replace('-POSDW_-E1BP', '') + '.csv', df.to_csv(index=False))
