# Importing Required Modules 
import cv2,numpy as np,os,pywhatkit,time,smtplib

# Loading Haarcascade face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# functions for face extraction
def face_extractor(img):
    # Function will detect faces and return the cropped face
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop the face found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Start Webcam
cap = cv2.VideoCapture(0)
count = 0

# Collect 100 samples of face from webcam
while True:

    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count += 1
        face = cv2.resize(face_extractor(frame), (200, 200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Save file in specified directory
        file_name_path = './faces/' + str(count) + '.jpg'
        cv2.imwrite(file_name_path, face)

        # Put count on images and display live count
        cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
        
    else:
        print("Face not found")
        pass

    if cv2.waitKey(1) == 13 or count == 100:
        break
        
cap.release()
cv2.destroyAllWindows()      
print("Collecting Samples Complete")

cap.release()

# importing training data we previously collected
data_path = './faces/'
onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

project_model  = cv2.face_LBPHFaceRecognizer.create()
# training model 
project_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi


# Opening Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    
    image, face = face_detector(frame)
    
    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        # Pass face to prediction model
        # results comprises of a tuple containing the label and the confidence value
        results = project_model.predict(face)
        
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'
            
        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        
        if confidence > 85:
            cv2.putText(image, "Hey Sonu", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
            msg = "Hey There, This is a AI generated message"
            pywhatkit.sendwhatmsg_instantly(phone_no="+91**",message=msg)
            print("Whatsapp Message sent Successfully!!")
            
            
            # Python code for Sending mail
            time.sleep(10)

            # creates SMTP session
            s = smtplib.SMTP('smtp.gmail.com', 587)

            # start TLS for security
            s.starttls()

            # Authentication (email , passwd)
            # sender email and passwd
            s.login(senders-gmail, password)

            # message to be sent
            message = "Hii, This is sonu"

            # sending the mail (sender email, receiver email , message)
            s.sendmail("sonumurmu092@gmail.com", "sonumurmu0923@gmail.com", message)

            # terminating the session
            s.quit()

            print("Mail sent successfully")
            
            break
         
        else:
            1
            os.system("aws ec2 run-instances  --image-id (ami-*****) --instance-type (t2.micro)  --subnet-id (subnet-****)  --count 1 --security-group-ids (sg-***)   --key-name (ec2-key)  > ec2.txt")
            print("Instance Launched")
            os.system("aws ec2 create-volume --availability-zone ap-south-1a --size 5 --volume-type gp2 --tag-specification  ResourceType=volume,Tags=[{Key=face,Value=volume}]  > ebs.txt")
            print("Volume Created")
            print("Please wait for 2 minutes instance is initializing")
            time.sleep(120)
            ec2_id = open("ec2.txt", 'r').read().split(',')[3].split(':')[1].split('"')[1]
            ebs_id = open("ebs.txt", 'r').read().split(',')[6].split(':')[1].split('"')[1]
            os.system("aws ec2 attach-volume --instance-id   " + ec2_id +"  --volume-id  " + ebs_id  +"  --device /dev/xvdf")
            print("Volume Successfully attached to the instance")
            break

    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
        
    if cv2.waitKey(1) == 13:
        break
        
cap.release()
cv2.destroyAllWindows()
