import xml.etree.ElementTree as ET
import xmltodict
import json
import os

folder = 'D:/XML_Data'
output = 'D:/Output'
files = [file for file in os.listdir(folder) if file.endswith('.xml')]

for f in files:
    xmltree = ET.parse(os.path.join(folder, f))
    xmlroot = xmltree.getroot()
    xmlstr = ET.tostring(xmlroot, encoding='utf8', method='xml')
    try:
        xmldata = xmltodict.parse(xmlstr)
    except:
        print('Error parsing xml '+f)
    else:
        with open(os.path.join(output, f.split('.')[0]+'.json'), 'w+') as json_file:
            json.dump(data_dict, json_file, indent=4, sort_keys=True)
