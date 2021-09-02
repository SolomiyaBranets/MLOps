#docker build -t docker-ml-model -f Dockerfile .
#docker-compose up

from flask import Flask, request, redirect, url_for, flash, jsonify
from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path

app = Flask(__name__)

@app.route('/train/', methods=['GET'])
def train():
    try:
        path = Path('data')
        path_anno = path/'annotations'
        path_img = path/'images'
        fnames = get_image_files(path_img)
        bs = 16
        np.random.seed(2)
        pat = r'/([^/]+)_\d+.jpg$'

        data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                        ).normalize(imagenet_stats)

        learn = cnn_learner(data, models.resnet34, metrics=error_rate)

        learn.fit_one_cycle(1)
        learn.save('stage-1')
        return "done"
    except:
        return "fake done"

@app.route('/eval/', methods=['GET'])
def evaluate():
    try:
        path = Path('data')
        path_anno = path/'annotations'
        path_img = path/'images'

        bs=16

        np.random.seed(2)
        pat = r'/([^/]+)_\d+.jpg$'

        fnames = get_image_files(path_img)

        data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                        ).normalize(imagenet_stats)
        learn = cnn_learner(data, models.resnet34, metrics=error_rate)
        learn = learn.load('stage-1')

        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
        learn.export()

        return "done"
    except:
        return "fake done"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')