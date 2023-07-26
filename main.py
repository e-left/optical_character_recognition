import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from train_knn_model import create_dataset, train_knn_model
from readtext import readtext
from contours import demo_contours

# demo contours
demo_contours()

# train classifier
dataset_filename = "dataset.pkl"
create_dataset_new = True
N1 = 2000
N2 = 2000
N3 = 2000
if create_dataset_new:
    create_dataset(dataset_filename, N1, N2, N3)
k1 = 2
k2 = 2
k3 = 3
display_metrics = True
model1, model2, model3 = train_knn_model(dataset_filename, k1, k2, k3, display_metrics)

# load testing data
image_filename = "/Users/eleft/Documents/School/digital_image_processing/assignment_2/text2_150dpi_rot.png"
text_filename = "/Users/eleft/Documents/School/digital_image_processing/assignment_2/text2.txt"

# read data
image = Image.open(image_filename)
image = np.array(image)
# remove black border
channels = 1
if len(image.shape) > 2:
    channels = image.shape[2]
white_color = 255 * np.ones((channels, ))
border = 5
image[:border, :] = white_color
image[:, :border] = white_color
image[:, - (border + 1):] = white_color
image[- (border + 1):, :] = white_color
# read target text
text = None
with open(text_filename, "r", encoding="utf-16") as f:
    text = f.readlines()
text = [x.replace("\n", "") for x in text]
text = [x.strip() for x in text]
text = "".join(text)
text = text.replace(" ", "")
text = [x for x in text]

# perform prediction
pred_text = readtext(image, (model1, model2, model3), (N1, N2, N3))

print("--- Predicted ---")
print(pred_text)

pred_text = pred_text.replace("\n", "").replace(" ", "")
pred_text = [x for x in pred_text]

conf_matrix = confusion_matrix(text, pred_text)
classes = sorted(list(set(text)))

sns.heatmap(conf_matrix, xticklabels=classes, yticklabels=classes)
plt.savefig("conf_matrix_in_text_2.png")
plt.show()

weighted_acc = balanced_accuracy_score(text, pred_text)
print(f"Weighted accuracy on test image: {weighted_acc}")