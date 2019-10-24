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

files = [file for file in os.listdir(folder) if file.endswith('.xml')]

for f in files:
    # print('\n'+f)
    xmltree = ET.parse(os.path.join(folder, f))
    xmlroot = xmltree.getroot()
    xmlstr = ET.tostring(xmlroot, encoding='utf8', method='xml')
    xmldata = xmltodict.parse(xmlstr)

    for table in tables:
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
            except:
                print(f + ' - Error converting to DataFrame for ' + table)
                # print(str(type(xmldoc)))          
            else:
                df['SourceFile'] = f
                # df.to_csv(os.path.join(output, f.split('.')[0] + table.replace('-POSDW_-E1BP', '') + '.csv.gz'), index=False, compression='gzip')
                df.to_json(os.path.join(output, f.split('.')[0] + table.replace('-POSDW_-E1BP', '') + '.json'), orient='records')
