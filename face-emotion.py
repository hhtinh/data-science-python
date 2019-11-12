import requests
import json

# set to your own subscription key value
subscription_key = '635ebf2bb65d4df58556ef1235685273'
assert subscription_key

# replace <My Endpoint String> with the string from your endpoint URL
face_api_url = 'https://southeastasia.api.cognitive.com/face/v1.0/'

image_url = 'https://upload.wikimedia.org/wikipedia/commons/3/37/Dagestani_man_and_woman.jpg'

headers = {'Ocp-Apim-Subscription-Key': subscription_key}

params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes': 'age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise',
}

response = requests.post(face_api_url, params=params,
                         headers=headers, json={"url": image_url})
print(json.dumps(response.json()))
