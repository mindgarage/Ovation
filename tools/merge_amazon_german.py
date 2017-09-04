import os
import json
import progressbar

indir = 'data/datasets/amazon_reviews_de/json'
outpath = 'data/datasets/amazon_reviews_de/reviews.txt'
files = os.listdir()

bar = progressbar.ProgressBar(max_value=len(files), redirect_stdout=True)
with open(outpath, 'w') as of:
    for f_i, json_file_name in enumerate(files):
        file_path = os.path.join(indir, json_file_name)
        with open(file_path, 'r') as jf:
            json_obj = json.load(jf)
            for json_review in json_obj:
                json_review_str = json.dumps(json_review, ensure_ascii=False)
                of.write(json_review_str + '\n')
        bar.update(f_i)
    bar.finish()