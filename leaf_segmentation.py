import cv2
import numpy as np
import time

from get_disease import get_disease

avg=[]

# I did this because 
k = cv2.waitKey(1)

def detect_diseases_camera(camNumber,vers_cascade,disease_th): 
    camera = cv2.VideoCapture(camNumber) # Hope there is only one camera module on board and the default is 0
    leaf_cascade = cv2.CascadeClassifier("cascades/v"+str(vers_cascade)+"/cascade.xml")    


    # Looping until it stops
    while True:
        ret, img = camera.read()
        

        if not ret:
            print("Camera not found")
            break
        key = cv2.waitKey(1)

        processed_image = detect_diseases(img,disease_th,leaf_cascade)
        if key == ord('q'):
            processed_image = False

        if processed_image is False:            
            print("Average fps: {}".format(np.mean(avg)))

            capture.release()
            cv2.destroyAllWindows()            

        # Saving the picture with boxes around sick leaves
        # And with the name as Year-Month-Day__Hour-Min-Sec.jpg

        # # I commented this option but it can be recommented and it works. 
        # # The folder 'unhealthy' needs to exists otherwise will raise an exception        
        # # This saves an image with the name as the current date - TimeOfDay, and 
        # # the diseases highlighted.

        elif processed_image is not None:            

            time_stamp = time.gmtime()
            file_name = time.strftime("%Y-%m-%d__%H-%M-%S",time_stamp)

            print(file_name)
            cv2.imwrite("unhealthy/"+file_name+".jpg",processed_image)    
        


def display(windowName,im,waitTime=0):
    cv2.imshow(windowName,im)
    cv2.waitKey(waitTime)


def get_mask(im):
    new_image = im.copy()
    copy = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
    # I got the HSV Values got from this image: 
    # https://i.stack.imgur.com/gyuw4.png
    #
    # In HSV color values start at 50, but we also include 
    # some white values because if the sun shines on the camera
    # it affects the detection (See the first few seconds in the 
    # testing video 1.mp4 and you will see how fast the fps
    # rises)

    lower_color = np.array([45, 15, 15], dtype=np.uint8) 
    upper_color = np.array([85, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(copy, lower_color, upper_color)

    new_image[mask == 0] = (0, 0, 0)    

    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)
    return gray

    
def detect_diseases_video(path,vers_cascade,disease_th):
    capture = cv2.VideoCapture(path)
    leaf_cascade = cv2.CascadeClassifier("cascades/v"+str(vers_cascade)+"/cascade.xml")    

    while(capture.isOpened()):
        _,img = capture.read()
        time_start = time.time()
        processed_image = detect_diseases(img,disease_th,leaf_cascade)
        if processed_image is False:            
            print("Average fps: {}".format(np.mean(avg)))

            capture.release()
            cv2.destroyAllWindows()            
        # Saving the picture with boxes around sick leaves
        # And with the name as Year-Month-Day__Hour-Min-Sec.jpg

        # # I commented this option but it can be recommented and it works. 
        # # The folder 'unhealthy' needs to exists otherwise will raise an exception        
        # # This saves an image with the name as the current date - TimeOfDay, and 
        # # the diseases highlighted.
        elif processed_image is not None:            
            
            time_stamp = time.gmtime()
            file_name = time.strftime("%Y-%m-%d__%H-%M-%S",time_stamp)
            print(file_name)
            cv2.imwrite("unhealthy/"+file_name+".jpg",processed_image)                              

def detect_diseases_image(path,vers_cascade,disease_th):
    img = cv2.imread(path)
    leaf_cascade = cv2.CascadeClassifier("cascades/v"+str(vers_cascade)+"/cascade.xml")
    detect_diseases(img,disease_th,leaf_cascade,False)


def detect_diseases(img,disease_th,leaf_cascade,video=True,useMask=True):    
    # ============================================================ #
    #   
    #   First of all, I haven't used "sick.mp4" yet. I tested it
    #   and I had to reconfig some settings but I didn't do it yet.
    #   However I saw that it still detected some diseases (Apple Scab - 
    #   which is what the disease in the picture is)
    #
    #   The disease detection works by resizing the image to a lower resolution
    #   (now 400ppx height - If you want to make it bigger you will also have to 
    #   change the maxXY depending on the size. I decided to go with maxXY = 35 
    #   by trial and error) but keeping the aspect ratio, and then detecting 
    #   as many leaves as possible using a cascade classifier (HAAS) trained
    #   on 1900 positive images and 800 negative images.
    #
    #   The positive images for this cascade were 100 pictures of 19 leaves, 
    #   pictures generated by rotating and placing the image on a negative 
    #   background, from an original image. (The original images
    #   and the background ones can be found at this link:
    #   ****
    #   
    #   
    #   We limit the detected leaves to 35x35 pixels, because anything bigger
    #   would be false-positives (e.g. detecting batches of leaves instead
    #   of individual leaves) 
    #   
    #   
    #   After we have each box, we then run a deep neural network
    #   model on each box trained in Keras on top of Tensorflow, which has been
    #   trained with transfer learning on the MobileNet model (small
    #   model, useful for small devices like RPi, Smartphones,etc,
    #   and with small disk space) on which I got around 99% accuracy
    #   for each disease on both train and validation. 
    #       
    #
    #   
    #   
    #   Some explaining for the arguments:   
    #
    #   img = the image for which we want to do the diseases
    #   disease_th = the threshold at which a leaf is considered sick
    #                for example, if disease_th is 4, if 4 diseases of
    #                a type are detected on a leaf, the leaf is considered
    #                sick
    #   leaf_cascade = a loaded cascade with which we will recognise the
    #                  leaves
    #   video (True) = If the image we recieve is from an image or not
    #   useMask (True) = If we apply the cascade classifier on a masked
    #                    image. It is recomended to do so to be sure there
    #                    are not that many false positives, and also because
    #                    the classifier works faster this way.
    #
    #   >>> returns: 
    #             If "video" is true:   
    #               - None if there are no sick leaves on the screen
    #               - An image (np.ndarray) if there are sick leaves
    #                 on the screen, with bounding boxes around each
    #                 unhealthy leaves (green) and bounding boxes
    #                 around each disease (red)
    #               - False if we have reached the end of video file.
    #
    #             If "video" is false:
    #               - Returns nothing, it just displays the image
    #                 along with the bounding boxes around each unhealthy
    #                 leaf and around each disease.
    #             
    # 
    # ============================================================ #            

    # 
    # We will have 3 images, the original (img), a copy of the original (original_copy) 
    # and a masked one (masked_image)
    # 
    # I didn't think of any way to make it with 2, which I assume would decrease
    # the time execution for a bit.

    # time_start is used to get the fps.
    time_start = time.time()   
    
    # if img is none that means we have reached the end of file.
    if img is None:
        return False

    
    if img.shape[1] > 400:
        newHeight = 400        
        newWidth = int(img.shape[1]*newHeight/img.shape[0])        
        img = cv2.resize(img, (newWidth, newHeight))    
        
    # If the video resolution is too small we exit.
    # Another idea it would be to upscale the image to 400, but that would make the
    # Classifier less precise (because of pixel noise)
    elif img.shape[1] < 400:
        return False
    maxXY = 35

    # Creating a copy in RGB because we will apply the model on this
    # And we will pass this image to get_disease.py
    # Keras reads the image in RGB using PIL, but cv2 reads it in BGR
    # Although there might not be big differences in results, it's safer to
    # be consistent.
    original_copy = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 


    # Creating a mask in order to extract only the leaves from the image.    
    masked_image = get_mask(img)                
    # display("Mask",masked_image) # If you want to visualize the mask uncomment this
    

    # We check if the image we recieved is a video frame or is just a simple image
    if video:                

        # I solved the FPS problem by limiting the maxSize of the leaves to something 
        # smaller, like 35,35, but it is dependent on the image height. It also works 
        # better now because it detects more individual leaves and less areas with 
        # batches of leaves (There are still some of those but significantly less)
        # Because of this I also got rid of the mean box.
        
        # Some things to note:
        # higher scaleFactor = less objects detected but it is faster
        # If you want accuracy, lower the scale factor but that will increase
        # the time it takes to process everything
        # If the video contains multiple shots of the same leaf, then from my experience 
        # a value of 1.25-1.30 is enough to be sure most of the leaves are checked.

        # A value of 1.30 with 3 minNeighbors gave an average of 7.39 FPS on video 1.mp4, and 
        # almost (if not all) all leaves were checked

        leaves = leaf_cascade.detectMultiScale(masked_image, 
                                               scaleFactor=1.25,
                                               minNeighbors=5,
                                               maxSize=(maxXY,maxXY))                
        
        if len(leaves) <= 0:        
            return

        isAnySick = False

        for rect in leaves:        
            x,y,w,h = rect
            per = w+h

            # Drawing the rectangle around the leaf. This line below can be commented, it is
            # purely for visualisation.
            # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)                                 

            # Each argument here is explained in the little documentation
            # at the beginning of the function            
            sick = get_disease(original_copy[y:y+h,x:x+w],masked_image[y:y+h,x:x+w])

            # If the leaf is sick ...            
            if bool(sick):                
                for cl in sick:
                    # ... we check if it has at least (disease_th) diseases of a type
                    # If we check only for one disease, we might get some false-positives 
                    # around the leaves or the area around the stalk.
                    numberDiseases = sick[cl][0][0]

                    # disease_th is default 2
                    if numberDiseases >= 1:                 
                        isAnySick = True   
                        # We use an iter because we want to skip the first value, which
                        # is the total amount of diseases of a type                                                                    
                        iteration = iter(sick[cl])
                        next(iteration)                    

                        # We draw a rectangle around the unhealthy leaf and later around the disease(s)                        
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                        # here "_s" comes from sick, so x_s is the x coordinate of the disease,etc.                  
                        for x_s,y_s,w_s,h_s in iteration:                                                        
                            x_s = x + x_s
                            y_s = y + y_s                        

                            cv2.rectangle(img,(x_s,y_s),(x_s+w_s,y_s+h_s),(0,0,255),1)             
                            cv2.putText(img,cl,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
        
        # I printed the FPS in the console because I wanted the image 
        # to be clean without text over the image (Besides the disease name)        
        fps = 1/(time.time()-time_start)
        avg.append(fps)
        display("Video feed",img,1)            

        # If there were any diseases detected we return the image so we can save it.
        # if isAnySick:
        #     return img
        # else:
        #     return None


    


    else:

        leaves = leaf_cascade.detectMultiScale(masked_image, scaleFactor=1.05,minNeighbors=2)                

        for rect in leaves:        
            x,y,w,h = rect

            sick = get_disease(original_copy[y:y+h,x:x+w],masked_image[y:y+h,x:x+w])
            if bool(sick):                
                for cl in sick:
                    numberDiseases = sick[cl][0][0]

                    if numberDiseases >= disease_th:                                                                             
                        iteration = iter(sick[cl])
                        next(iteration)                    

                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
              
                        for x_s,y_s,w_s,h_s in iteration:                                                        
                            x_s = x + x_s
                            y_s = y + y_s                        
                            cv2.rectangle(img,(x_s,y_s),(x_s+w_s,y_s+h_s),(0,0,255),1)             
                            cv2.putText(img,cl,(x-10,y-10),cT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)
        
        cv2.imshow("Image",img)    
        k = cv2.waitKey(0)
        if k == ord('q') or k == ord('Q'):
            return