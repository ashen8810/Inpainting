import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours
import matplotlib.pyplot as plt


drawing=False

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def parse_mask(mask):
    ret,mask = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    return mask

def run(imageName,output,iterate):
    import cv2

    #iterating process
    if iterate == False:
        imageName = "results/"+imageName
    print(imageName)

    pt1_x , pt1_y = None , None
    def save_image(image):
        cv2.imwrite("results/image.jpg",image)


    # mouse callback function
    def line_drawing(event,x,y,flags,param):

        global pt1_x,pt1_y,drawing

        if event==cv2.EVENT_LBUTTONDOWN:
            drawing=True
            pt1_x,pt1_y=x,y

        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,255,0),thickness=8)
                pt1_x,pt1_y=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(0,255,0),thickness=8)  
        if cv2.waitKey(1) & 0xFF == 27:
            save_image(img)
                
    
    create_dir("results")
    img = cv2.imread(imageName)
    cv2.namedWindow('test draw')
    cv2.setMouseCallback('test draw',line_drawing)

    while(1):
        cv2.imshow('test draw',img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            save_image(img)
            break
            
    cv2.destroyAllWindows()


    images = ['results/image.jpg']
    masks = ['results/mask.png']
    image = plt.imread(images[0])

    # Extract the specific color
    target_color = np.array([0, 255, 0])  # Green color in RGB format
    color_threshold = 80  # Adjust the threshold for color matching

    # Calculate the color difference
    color_difference = np.abs(image - target_color)

    # Create a mask for the pixels within the color threshold
    mask = np.all(color_difference < color_threshold, axis=2)

    # Apply the mask to the original image
    extracted_color = np.zeros_like(image)
    extracted_color[mask] = image[mask]

    import cv2
    mask = np.asarray(mask,dtype=np.int64)
    cv2.imwrite(masks[0],extracted_color)



    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        bboxes = mask_to_bbox(y)
        width = x.shape[0]
        height = x.shape[1]

        for bbox in bboxes:

            X,Y,Xh,Yh = bbox    

            image = x[Y:Yh,X:Xh]
           

            new_width = int(x.shape[0]*2)
            new_height = int(x.shape[1]*2)

            #interpolation
            interpolation_method = cv2.INTER_NEAREST  # You can choose from other methods like INTER_NEAREST, INTER_CUBIC, etc.
            
            
            try:
                resized_image = cv2.resize(cv2.imread("results/"+output), (new_width, new_height), interpolation=interpolation_method)
            except:
                resized_image = cv2.resize(cv2.imread(imageName), (new_width, new_height), interpolation=interpolation_method)

            
            
            resized_mask = cv2.resize(parse_mask(y), (new_width, new_height), interpolation=interpolation_method)
            # resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
  


            # resized_mask = cv2.GaussianBlur(resized_mask, (5, 5), 3)

            roi = resized_image[Y:Yh,X:Xh]
            roi = cv2.bilateralFilter(roi, d=6, sigmaColor=75, sigmaSpace=75)
            # roi = cv2.medianBlur(roi,9)
            resized_image[Y:Yh,X:Xh] =roi


            #Inpainting
            inpainting_result = cv2.inpaint(resized_image, resized_mask,3.8,cv2.INPAINT_TELEA)

            inpainting_result = cv2.resize(inpainting_result, (x.shape[1], x.shape[0]))

            # inpainting_result = cv2.cvtColor(inpainting_result, cv2.COLOR_BGR2RGB)
            try:
           
                cv2.imshow('Signatures',cv2.imread(imageName)[Y-10:Yh+10,X-10:Xh+10])
            except:
                cv2.imshow('Signatures',cv2.imread(imageName)[Y:Yh,X:Xh])

            #unsharp masking
            bl = cv2.bilateralFilter(inpainting_result[Y:Yh,X:Xh], d=5, sigmaColor=75, sigmaSpace=75)
            m = cv2.subtract(inpainting_result[Y:Yh,X:Xh],bl)
            inpainting_result[Y:Yh,X:Xh] = cv2.add(inpainting_result[Y:Yh,X:Xh],cv2.multiply(m,0.95))


            cv2.imwrite("results/"+output,inpainting_result)
        
        
        
        cv2.imshow('Result Image',inpainting_result)
        cv2.imshow('Mask',parse_mask(y))


        cv2.imshow('Original Image', cv2.imread(imageName))

     
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == '__main__':
    run("12.jpg","12.jpg",1)
