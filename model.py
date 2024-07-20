from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np



def get_class(model, labels, image):

  np.set_printoptions(suppress=True)

  model = load_model(model, compile=False)

  class_names = open(labels, "r").readlines()

  data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

  image = Image.open(image).convert("RGB")

  size = (224, 224)

  image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

  image_array = np.asarray(image)

  normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

  data[0] = normalized_image_array

  # Przewiduje model

  prediction = model.predict(data)

  index = np.argmax(prediction)

  class_name = class_names[index]

  confidence_score = prediction[0][index]


  bird = class_name[2:].strip()

  if bird == 'Sparrow':

    msg = "Oto ciekawostka o danym obiekcie:Wróble konstruują gniazda o dosyć charakterystycznym owalnym kształcie. Do gniazda na ogół prowadzi jeden otwór, przez który matka wlatuje, by doglądać swoje maleństwa. W terenach zabudowanych najczęstszymi miejscami wybieranymi przez samice są przerwy dylatacyjne, szczeliny, poddasza lub niewielkie przestrzenie, które zagwarantują gniazdom stabilizację."

  elif bird == 'Pigeon':

    msg = "Półtora miesiąca później, zagubiony ptak odnalazł się na Vancouver Island w Kanadzie, czyli pokonał 8 tysięcy kilometrów! Dla porównania dodamy, że przeciętne gołębie pocztowe latają na trasach około 650 km. Najlepsze ptaki potrafią przelecieć jednorazowo około 1000 km."

  return bird, confidence_score, msg



