import json

samples = []

def save_to_array(datapath, labely):
    with open(datapath, 'r', encoding="utf8") as fin1:
        data1 = eval(fin1.read())
        for index, item in enumerate(data1):
            id = item['id']
            label = item['label']
            type = item['type']
            a = item['a']
            b = item['b']
            c = item['c']
            d = item['d']
            e = item['e']
            f = item['f']

            samples_b = dict()
            samples_b['id'] = id
            samples_b['label'] = labely
            samples_b['type'] = type
            samples_b['a'] = a
            samples_b['b'] = b
            samples_b['c'] = c
            samples_b['d'] = d
            samples_b['e'] = e
            samples_b['f'] = f

            samples.append(samples_b)

    return samples

samples1 = save_to_array('Path/to/TNT.json', 0)

samples2 = save_to_array('Path/to/TNT.json', 1)

with open('path_to_all.json', 'w', encoding='utf-8') as outfile:
    # json.dump(samples1, outfile)
    json.dump(samples2, outfile)
    print('new json is writing...')
    outfile.write('\n')

print('dataset processing finished...')