import RPi.GPIO as GPIO
import time
from picamera import PiCamera
import requests
import Adafruit_DHT
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

GPIO.setmode(GPIO.BCM)

# Set sensor type : Options are DHT11,DHT22 or AM2302
sensor = Adafruit_DHT.DHT11


# initialize ultrasonic sensor GPIO pins
GPIO_TRIGGER = [26, 16, 6, 0]
GPIO_ECHO = [20, 19, 12, 1]

# initialize LED and  GPIO pins
GPIO_LED = [21, 13, 5, 7]

# Set GPIO pin number (use BCM GPIO numbering)
hum_pin = 2

# set up GPIO pins
for i in range(4):

    GPIO.setup(GPIO_TRIGGER[i], GPIO.OUT)

    GPIO.setup(GPIO_ECHO[i], GPIO.IN)

    GPIO.setup(GPIO_LB[i], GPIO.OUT)


BOT_TOKEN = 'ABCD'
CHAT_ID = 'ABCDEF'

# initialize camera
camera = PiCamera()
camera.resolution = (224, 224)

# load TensorFlow Lite model
model_path = 'animal_classification_model.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def measure_distance(trigger_pin, echo_pin):

    # set trigger pin to high for 10us
    GPIO.output(trigger_pin, True)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, False)

    pulse_start = time.time()
	
    # measure echo pulse duration
    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()
    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()
		
    # calculate distance in cm
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
	
    return distance
	
def sendImage(path,corner):

    # open the image file
    with open(path, 'rb') as f:
        # send a POST request to the Telegram Bot API with the image as multipart/form-data
        url = f'https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto'
        response = requests.post(url, data={'chat_id': CHAT_ID, 'caption':'object detected at side '+str(corner) }, files={'photo': f})


    # check if the request was successful

    if response.status_code == 200:

        print('Image sent successfully.')

    else:

        print('Error sending image:', response.text)

# classify image
def classify_image(image):
    # pre-process image
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = image.astype('float32')

    # set input tensor
    input_tensor_index = input_details[0]['index']
    interpreter.set_tensor(input_tensor_index, image)

    # run model
    interpreter.invoke()

    # get output tensor
    output_tensor_index = output_details[0]['index']
    output_tensor = interpreter.get_tensor(output_tensor_index)

    # return class label
    if len(output_tensor[0]) == 1:
        if output_tensor[0][0] > 0.6:
            return 'animal'
        else:
            return 'not_animal'
    else:
        if output_tensor[0][0] > output_tensor[0][1]:
            return 'not_animal'
        else:
            return 'animal'

def humidity_sens():
    # Check if reading was successful

    if humidity is not None and temperature is not None:
        print('Temperature={0:0.1f}*C  Humidity={1:0.1f}%'.format(temperature, humidity))
     
        # Check if humidity is below threshold

        if humidity < threshold_humidity:

            # Send alert to Telegram bot
            telegram_message = f'Humidity is below threshold: {humidity}% And Temperature={temperature}*C'
            telegram_api_url = f'https://api.telegram.org/bot{telegram_bot_token}/sendMessage'
            telegram_params = {'chat_id': telegram_chat_id, 'text': telegram_message}
            response = requests.post(telegram_api_url, params=telegram_params)
            if response.ok:
                print('Alert sent to Telegram bot')
            else:
                print('Failed to send alert to Telegram bot')
    else:
        print('Failed to get reading from sensor.')


try:
    while True:
        for i in range(4):
            # measure distance using ultrasonic sensor
            distance = measure_distance(GPIO_TRIGGER[i], GPIO_ECHO[i])
            
            # check if object is within range
            if distance < 27:

                # turn on LED for corresponding corner
                GPIO.output(GPIO_LB[i], True)
                print(f"Sensor {i+1}:{distance} cm")
                time.sleep(0.2)
                GPIO.output(GPIO_LB[i], False)

                # capture image using camera
                filename = f"corner_{i+1}.jpg"
                camera.capture(filename)


                load image and classify
                image = Image.open(filename).resize((224, 224))
                image = np.array(image)
                class_label = classify_image(image)
                print("Obejct is ",class_label)
                # produce sound if image contains an animal
                if class_label == 'animal':
                    GPIO.output(BUZZER_PIN, True)
                    time.sleep(0.5)
                    GPIO.output(BUZZER_PIN, False)
                    
                    # send message with image to Telegram
                    sendImage(filename,i+1)
                    
                    #send_alert_message_with_image(filename)
                    print(filename,"sent")

            else:
                # turn off LED for corresponding corner
                GPIO.output(GPIO_LB[i], False)

except KeyboardInterrupt:
    print("Program stopped by user")
    GPIO.cleanup()

finally:
    # release camera resources
    camera.stop_preview()
    camera.close()
