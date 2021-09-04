from flask import Flask, request, redirect, url_for, flash, jsonify
from fastai.vision import *
from fastai.metrics import error_rate
from pathlib import Path
import mlflow.fastai
from mlflow import log_artifact
from tensorboardX import SummaryWriter
from TensorBoardCallback import *


app = Flask(__name__)

@app.route('/train/', methods=['GET'])
def train():
    #training code from the course 
    path = Path('data')
    path_anno = path/'annotations'
    path_img = path/'images'
    fnames = get_image_files(path_img)
    bs = 16
    np.random.seed(2)
    pat = r'/([^/]+)_\d+.jpg$'
    data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs
                                    ).normalize(imagenet_stats)
    #integrate tensorboard
    writer = SummaryWriter(comment='test')
    mycallback = partial(TensorBoardFastAI, writer, track_weight=True, track_grad=True, metric_names=['error rate'])
    
    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    learn.callback_fns.append(mycallback)

    #integrate mlflow
    mlflow.fastai.autolog()
    
    #train
    with mlflow.start_run() as run:
        learn.fit_one_cycle(4)
    learn.save('stage-1')

    #evaluate
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
    
    #save the model as pkl
    learn.export()
    #save the model to mlflow
    log_artifact("data/images/export.pkl")

    return run.info.run_id


@app.route('/predict/', methods=['POST'])
def predict():
    # get image name
    data = request.get_json()   
    path = Path('data')

    #load model
    learn = load_learner(path/'images')

    #get predictions 
    prediction = learn.predict(open_image(Path('data/new_images/'+data)))
    category = prediction[0].obj
    probability = round(float(prediction[2][0]), 5)
    return jsonify({"path": data, "category": category, "probability" : probability})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')