# import the necessary packages
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image,ImageOps
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

classes=['axes','boots','carabiners','crampons','gloves','hardshell_jackets','harnesses','helmets','insulated_jackets','pulleys','rope','tents']

def load_our_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    model = load_model('model.h5')
    model._make_predict_function()

def prepare_image(im, size=128, fill_color=(255,255,255)): #take a PIL image
    im.thumbnail((size,size), Image.ANTIALIAS)
    x, y = im.size    
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    new_im=ImageOps.equalize(new_im)
    im_array = img_to_array(new_im)
    im_array=np.expand_dims(im_array, axis=0)
    #imagenet_utils.preprocess_input(im_array)
    return im_array

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            preds = model.predict_classes(image)
            #results = imagenet_utils.decode_predictions(preds)
            data["prediction"] = classes[int(preds)]

            # loop over the results and add them to the list of
            # returned predictions
            #for (imagenetID, label, prob) in results[0]:
            #    r = {"label": label, "probability": float(prob)}
            #    data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_our_model()
    app.run(port=8080, host='0.0.0.0')

