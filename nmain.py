from telegram import Update, InputFile
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from tensorflow.keras.models import load_model
from norvig import solve
import norvig
import numpy as np
import imutils
import cv2
import pickle

# pathImage="puzzle.jpg"

def disp(img,numbers,color = (0,255,0)):
    width = int(img.shape[1]/9)
    hei = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                cv2.putText(img, str(numbers[(y*9)+x]),(x*width+int(width/2)-10, int((y+0.8)*hei)), cv2.FONT_HERSHEY_COMPLEX_SMALL,2, color, 2, cv2.LINE_AA)
    return img

# Function to solve the Sudoku puzzle
def solve_sudoku(pathImage):
    # Your existing script logic
    # Replace this with the actual function call to your Sudoku solver
    # This is a placeholder that copies the input image to output

    heightImg=450
    widthImg=450

    # HTTP API:7837745480:AAH8W8DMJT6fl0qAM6lCKFEMfzP2hMyNDqA

    # preprocessing image

    img=cv2.imread(pathImage)
    img= cv2.resize(img,(widthImg,heightImg))
    im1=img.copy()
    imgBlank = np.zeros((heightImg,widthImg,3),np.uint8)
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgblur=cv2.GaussianBlur(imggray,(5,5),1)
    imgthresh=cv2.adaptiveThreshold(imgblur,255,1,1,11,2)

    # finding the countours
    imgcountours=img.copy()
    countours,hierarchy=cv2.findContours(imgthresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgcountours,countours,-1,(0,255,0),3)
    imgcountours=cv2.cvtColor(imgcountours,cv2.COLOR_BGR2GRAY)


    # finding the biggest countour
    biggest = np.array([])
    max_area = 0
    for i in countours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area

    # reordering the points
    biggest = biggest.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), dtype=np.int32)
    add = biggest.sum(1)
    myPointsNew[0] = biggest[np.argmin(add)]
    myPointsNew[3] =biggest[np.argmax(add)]
    diff = np.diff(biggest, axis=1)
    myPointsNew[1] =biggest[np.argmin(diff)]
    myPointsNew[2] = biggest[np.argmax(diff)]
    biggest=myPointsNew

    # warp perspective
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)



    # split image and find all digit available
    imgSolvedDigit=imgBlank.copy()
    rows=np.vsplit(imgWarpGray,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            boxes.append(box)

    pickle_in = open("model_trained.h5","rb")
    model = pickle.load(pickle_in)


    result = []
    for image in boxes:
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        ## GET PREDICTION
        predictions = model.predict(img)
        classIndex = np.argmax(predictions, axis=-1)
        probabilityValue = np.amax(predictions)
        
        ## SAVE TO RESULT
        if probabilityValue > 0.80:
            result.append(int(classIndex[0]))
            print(classIndex, probabilityValue)
        else:
            result.append(0)
            print(0)

    print(result)

    imgDetectedDigits=disp(imgBlank.copy(),result)
    res=np.asarray(result)
    posArrray=np.where(res>0,0,1)
    print(posArrray)

    # solving the sudoku

    string = ''.join(map(str, result))
    solved=solve(string)

    solution = [int(char) for char in list(solved.values())]
    fin_soln=np.array(solution)*posArrray
    # print(fin_soln)
    imgSolvedDigit=disp(imgSolvedDigit,fin_soln.tolist())


    # overlay the solution by inversing the warp perspective
    pts2 = np.float32(biggest) 
    pts1 =  np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
    matrix_new = cv2.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img.copy()
    imgInvWarpColored = cv2.warpPerspective(imgSolvedDigit, matrix_new, (widthImg, heightImg))
    inv_perspective = cv2.addWeighted(imgInvWarpColored, 1, im1, 0.5, 1)
    solved_image_path = "solved_puzzle.jpg"
    # Call your solving logic and save the result as 'solved_puzzle.jpg'
    cv2.imwrite(solved_image_path, inv_perspective)
    return solved_image_path