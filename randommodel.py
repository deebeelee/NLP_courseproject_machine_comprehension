import random
import json

class RandomModel():
    def __init__(self):
        self.model=None

    def predict(self,fpath, outpath):
        pred_dict = {}
        with open(fpath,'r') as fp:
            test_set = json.load(fp)
            for title in test_set["data"]:
                for paragraph in title["paragraphs"]:
                    for question in paragraph["qas"]:
                        pred_dict[question["id"]] = random.randint(0,1)
        with open(outpath,'w') as outp:
            json.dump(pred_dict,outp)
        return