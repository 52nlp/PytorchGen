from flask import Flask, request
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json

from torchvision import models, transforms
from torch.autograd import Variable
import torchvision.models as models
from gen_data_smote import gen_fake_data_smote
from gen_data_wgan import gen_fake_data
from fraud_classifier import train_classifier, eval_test_set

from PIL import Image

import requests

# All the 1000 imagenet classes
class_labels = 'imagenet_classes.json'

# Read the json
with open('imagenet_classes.json', 'r') as fr:
	json_classes = json.loads(fr.read())

app = Flask(__name__)

# Allow 
CORS(app)

# Path for uploaded images
UPLOAD_FOLDER = 'data/uploads/'

# Allowed file extransions
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def hello():
	return "Hello World!"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		print("request data", request.data)
		print("request files", request.files)

		# check if the post request has the file part
		if 'file' not in request.files:
			return "No file part"
		file = request.files['file']

		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			
			# Send uploaded image for prediction
			predicted_image_class = predict_img(UPLOAD_FOLDER+filename)
			print("predicted_image_class", predicted_image_class)

	return json.dumps(predicted_image_class)

@app.route('/smote', methods=['GET', 'POST'])
def gen_SMOTE_data():
    csv_file_by_SMOTE = gen_fake_data_smote()
    print("gen_SMOTE_data() is called")
    return json.dumps("Sucess")

@app.route('/gan', methods=['GET', 'POST'])
def gen_wgan_data():
    csv_file_by_wgan = gen_fake_data(20000)
    print("gen_wgan_data() is called")
    return json.dumps("Sucess")

@app.route('/smoteclassifier', methods=['GET'])
def run_smote_classifier():
    aug_file = "./fake_data/fraud/smote_2019-09-08_100_5_30000.csv"
    aug_model, testloader = train_classifier(augment_file=aug_file, n_samples=20000)
    model_acc, model_recall, model_precision, model_f1, model_cm = eval_test_set(aug_model, testloader)
    ret_data = { 'acc':model_acc, 'recall':model_recall, 'precision':model_precision, 'f1':model_f1 }

    print("run_smote_classifier() is called")
    return json.dumps(ret_data)

@app.route('/ganclassifier', methods=['GET'])
def run_gan_classifier():
    aug_file = "./fake_data/fraud/wgan_2019-09-08_30000.csv"
    aug_model, testloader = train_classifier(augment_file=aug_file, n_samples=20000)
    model_acc, model_recall, model_precision, model_f1, model_cm = eval_test_set(aug_model, testloader)
    ret_data = { 'acc':model_acc, 'recall':model_recall, 'precision':model_precision, 'f1':model_f1 }

    print("run_ZetaGAN_classifier() is called")
    return json.dumps(ret_data)


def predict_img(img_path):

	# Available model archtectures = 
	#'alexnet','densenet121', 'densenet169', 'densenet201', 'densenet161','resnet18', 
	#'resnet34', 'resnet50', 'resnet101', 'resnet152','inceptionv3','squeezenet1_0', 'squeezenet1_1',
    #'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn','vgg19_bn', 'vgg19'
	

	# Choose which model achrictecture to use from list above
	architecture = models.squeezenet1_0(pretrained=True)
	architecture.eval()

	# Normalization according to https://pytorch.org/docs/0.2.0/torchvision/transforms.html#torchvision.transforms.Normalize
	# Example seen at https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                                     std=[0.229, 0.224, 0.225])
	    
	# Preprocessing according to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
	# Example seen at https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L89-L101

	preprocess = transforms.Compose([
	   transforms.Resize(256),
	   transforms.CenterCrop(224),
	   transforms.ToTensor(),
	   normalize
	])

	# Path to uploaded image
	path_img = img_path

	# Read uploaded image
	read_img = Image.open(path_img)

	# Convert image to RGB if it is a .png
	if path_img.endswith('.png'):
	    read_img = read_img.convert('RGB')

	img_tensor = preprocess(read_img)
	img_tensor.unsqueeze_(0)
	img_variable = Variable(img_tensor)

	# Predict the image
	outputs = architecture(img_variable)

	# Couple the ImageNet label to the predicted class
	labels = {int(key):value for (key, value)
	          in json_classes.items()}
	print("\n Answer: ",labels[outputs.data.numpy().argmax()])


	return labels[outputs.data.numpy().argmax()]


if __name__ == "__main__":
	app.run(debug=True)
