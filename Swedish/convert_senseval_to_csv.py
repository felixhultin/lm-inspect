import re
import pandas
import xml.etree.ElementTree as ET

def convert_xml_to_csv(path):
    root = ET.parse(path).getroot()
    data = []
    for instance in root.iter('instance'):
        sense_key = instance.find('answer').attrib.get('senseid')
        word = sense_key.split("_")[0]
        context = instance.find('context')
        content = ET.tostring(context, encoding='unicode')
        tokens = content.strip("<context>").split("</context>")[0].split()
        pos, tag = next( (idx, t) for idx, t in enumerate(tokens) if t.startswith("<head>"))
        tokens[pos] = tag.split("<head>")[1].split("</head>")[0]
        text = " ".join(tokens)
        data.append([sense_key, None, pos, text])
    pandas.DataFrame(data, columns = ['sense_key', 'lemma', 'pos', 'text']).to_csv(path + ".csv", sep="\t", header=False, index=False)



if __name__ == '__main__':
    convert_xml_to_csv('senseval/swedish_lexical_sample_TRAIN_corpus.xml')
    convert_xml_to_csv('senseval/swedish_lexical_sample_GOLD_corpus.xml')

    pass
