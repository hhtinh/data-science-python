import xml.etree.ElementTree as ET
import xmltodict
import json
import os
import pandas as pd
from collections import OrderedDict

folder = '../XML_Data'
output = '../Upload'

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

def extract_data(table_name):

    files = [file for file in os.listdir(folder) if file.endswith('.xml')]
    for f in files:
        # print('\n'+f)
        xmltree = ET.parse(os.path.join(folder, f))
        xmlroot = xmltree.getroot()
        xmlstr = ET.tostring(xmlroot, encoding='utf8', method='xml')
        xmldata = xmltodict.parse(xmlstr)

        try:
            xmldoc = xmldata["_-POSDW_-POSTR_CREATEMULTIPLE04"]["IDOC"]["_-POSDW_-E1POSTR_CREATEMULTIP"][table_name]
        except:            
            pass # print('No data for ' + table)
        else:
            if type(xmldoc) is OrderedDict:
                tmp = []
                tmp.append(xmldoc)
                xmldoc = tmp
            try:
                df_tmp = pd.DataFrame(xmldoc)
                df_tmp['SourceFile'] = f
                global df                
                df = df.append(df_tmp, ignore_index=True, sort=False)                
            except:                
                print(f + ' - Error converting to DataFrame for ' + table_name)
                print(str(type(xmldoc)))
            
            print(table_name + ' | Rows: ' + str(df.shape[0]) + ', Columns: ' + str(df.shape[1]))

    df.to_csv(os.path.join(output, table_name.replace('_-POSDW_-E1BP', '') + '.csv'), index=False)
    # df.to_json(os.path.join(output, table_name.replace('_-POSDW_-E1BP', '') + '.json'), orient='records')

    df = pd.DataFrame() # flush data

if __name__ == '__main__':
    df = pd.DataFrame()
    for table in tables:
        extract_data(table)
