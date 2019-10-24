import xml.etree.ElementTree as ET
import xmltodict
import json
import os

folder = '../XML_Data'
output = '../Output'
files = [file for file in os.listdir(folder) if file.endswith('.xml')]

for f in files:
    print(f)
    tree = ET.parse(os.path.join(folder, f))
    xml_data = tree.getroot()
    xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')
    data_dict = dict(xmltodict.parse(xmlstr))

    with open(os.path.join(output, f.split('.')[0]+'.json'), 'w+') as json_file:
        json.dump(data_dict, json_file, indent=4, sort_keys=True)
