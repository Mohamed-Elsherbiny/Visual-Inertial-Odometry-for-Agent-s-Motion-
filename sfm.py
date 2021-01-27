#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
class readimages:
    """
    A class used to read images from folder

    ...

    Attributes
    ----------
    img_gray : list
        a list of gray scale images
    img_rgb : list
        a list of rgb images
    size : tuble
        image size

    """
    def __init__(self,no,folder):
        """
            Parameters
            ----------
            no : int
                no of images to read
            folder : string
                folder path which contains the images
        """
        folders = glob.glob(folder)
        imagenames_list = []
        # get directory for each image 
        for folder in folders:
            for f in glob.glob(folder+'/*.png'):
                imagenames_list.append(f)
        
        read_images = [] 
        self.img_gray = []
        self.img_rgb = []
        # read images 
        for image in imagenames_list:
            read_images.append(cv2.imread(image))
            if len(read_images) > no:
                break
             
        # transfrom BGR to RGB
        height, width, layers = read_images[0].shape
        self.size = (width,height)
        self.img_rgb = [cv2.cvtColor(x, cv2.COLOR_BGR2RGB) for x in read_images]
        if layers > 1:
            self.img_gray = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in read_images]

class montage:
    """
    A class used to plot no of images in one image

    ...

    Attributes
    ----------
    imgs : array (2D or 3D)
        image containg all the input images

    """
    def __init__(self,images,size,scale=20):
        """
            Parameters
            ----------
            images : list
                list of images 
            size : array [a,b]
                a : Define number of columns, b : Define number of rows
            scale : int
                scale for images to fit in the diseried view (default = 20)
        """
        nrows = size[0] # Define number of columns
        ncols = size[1] # Define number of rows
        self.imgs = []
        resized = []
        scale_percent = scale # percent of original size
        for i in range(len(images)):
            width = int(images[i].shape[1] * scale_percent / 100)
            height = int(images[i].shape[0] * scale_percent / 100)
            dim = (width, height)
            resized.append(cv2.resize(images[i], dim, interpolation = cv2.INTER_AREA))
        images = resized
        
        
        layers = len(np.shape(images[0]))
        if layers > 2:
            image_heigt, image_width, layers = images[0].shape
            rgbArray = np.zeros((nrows*image_heigt,ncols*image_width,3), 'uint8')
            for a in range(nrows):
                if a*ncols+ncols > len(images):
                        break
                for b in range(ncols):
                    if a*ncols+b > len(images):
                        break
                    rgbArray[a*image_heigt:a*image_heigt+image_heigt,b*image_width:b*image_width+image_width, 0] = images[a*ncols+b][...,0]
                    rgbArray[a*image_heigt:a*image_heigt+image_heigt,b*image_width:b*image_width+image_width, 1] = images[a*ncols+b][...,1]
                    rgbArray[a*image_heigt:a*image_heigt+image_heigt,b*image_width:b*image_width+image_width, 2] = images[a*ncols+b][...,2]
                    
            self.imgs =  rgbArray
        else:
            image_heigt, image_width = images[0].shape
            pixels = np.zeros([nrows*image_heigt,ncols*image_width], dtype = np.uint8)
            for a in range(nrows):
                if a*ncols+ncols > len(images):
                        break
                for b in range(ncols):
                    if a*ncols+b > len(images):
                        break
                    pixels[a*image_heigt:a*image_heigt+image_heigt,b*image_width:b*image_width+image_width] = images[a*ncols+b][0:image_heigt,0:image_width]
            
            self.imgs =  pixels
            
class cameraParameters:
    """
    A class used as object stores the intrinsic, extrinsic, and lens distortion parameters of a camera.

    ...

    Attributes
    ----------
    IntrinsicMatrix : Mat (3x3)
    RadialDistortion : list [a,b, ....]
    TangentialDistortion : list [a,b]
    RotationVectors : Mat (3x3)
    TranslationVectors : list [a,b, ....]

    """
    def __init__(self, IntMat, Intval, RadDist, Radval, TangDist, Tangval, RotMat, Rotval, TranVec, Tranval):
        """
            Parameters
            ----------
            IntMat : String
                name
            Intval : Mat (3x3)
                IntrinsicMatrix 
            RadDist : String
                name
            Radval : list [a,b, ....]
                RadialDistortion 
            TangDist : String
                name
            Tangval : list [a,b]
                TangentialDistortion
            RotMat : String
                name
            Rotval : Mat (3x3)
                RotationVectors
            TranVec : String
                name
            Tranval : list [a,b, ....]
                TranslationVectors
        """
        self.IntrinsicMatrix = Intval
        self.RadialDistortion = Radval
        self.TangentialDistortion = Tangval
        self.RotationVectors = Rotval
        self.TranslationVectors = Tranval
import cv2 as cv
class Detect_and_match:
    """
    A class used Detect, compute and match interest points and their descriptors.

    ...

    Attributes
    ----------
    matchedPoints1 : list of tuple(points)
        coordinates of the matched features of image1
    matchedPoints2 : list of tuple(points)
        coordinates of the matched features of image2
    prevFeatures : list 
        list of feature matches for the first image
    currFeatures : list 
        list of feature matches for the first image
    

    """
    def __init__(self,detector_name,detector,image1,image2,ratio=0.75, nmatches=10, plot = False):
        """
            Parameters
            ----------
            detector : cv2.detector_name
            detector_name : String
                name of the detector
            image1 : array
                first image 
            image2 : array
                second image
            nmatches: int
                number of matches to draw on the image (default=10)
            plot: bool
                if you wnat to plot the result plot = True (default=False)
            ratio: float
                MaxRatio (default=0.75)
        """
        # Initialize lists
        self.matchedPoints1 = []
        self.matchedPoints2 = []
        kp1, des1 = detector.detectAndCompute(image1, None)
        kp2, des2 = detector.detectAndCompute(image2, None)
        self.prevFeatures = kp1
        self.currFeatures = kp2
        
        # create BFMatcher object
        if detector_name == 'ORB':
            print(type('detector'))
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            # Match descriptors.
            matches = bf.match(des1,des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            # Coordinates of the matched features 
            self.matchedPoints1 = [kp1[mat.queryIdx].pt for mat in matches] 
            self.matchedPoints2 = [kp2[mat.trainIdx].pt for mat in matches]
            # Draw first nmatches matches.
            self.img3 = cv.drawMatches(image1,kp1,image2,kp2,matches[:nmatches],None, flags=2)
        else :
            bf = cv.BFMatcher(cv.NORM_L1, crossCheck=False)
            # Match descriptors.
            matches = bf.knnMatch(des1,des2,k=2)
            # Apply ratio test
            coordinates = []
            good = []
            for m,n in matches:
                if m.distance < ratio*n.distance:
                    good.append([m])
                # Coordinates of the matched features 
                    coordinates.append(m)
                
                    
            self.matchedPoints1 = np.int32([kp1[mat.queryIdx].pt for mat in coordinates])
            self.matchedPoints2 = np.int32([kp2[mat.trainIdx].pt for mat in coordinates])
            # Draw first nmatches matches.a
            # cv.drawMatchesKnn expects list of lists as matches.
            self.img3 = cv.drawMatchesKnn(image1,kp1,image2,kp2,good[:nmatches],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        


        
        
        
        
        
        if plot == True:
            plt.figure(figsize=(16, 16))
            plt.title(type(detector))
            plt.imshow(self.img3); plt.show()

class helperEstimateRelativePose:
    """
    A class used to Robustly estimate relative camera pose.

    ...
    
    Attributes
    ----------
      orientation : the orientation of camera 2 relative to camera 1
                    specified as a 3-by-3 rotation matrix
      location    : the location of camera 2 in camera 1's coordinate system
                    specified as a 3-element vector
      inlierIdx   : the indices of the inlier points from estimating the
                    fundamental matrix
    

    """
    def __init__(self,matchedPoints1,matchedPoints2,cameraParams):
        """
            Parameters
            ----------
                matchedPoints1 : points from image 1 specified as an M-by-2 matrix of
                   [x,y] coordinates, or as any of the point feature types
                matchedPoints2 : points from image 2 specified as an M-by-2 matrix of
                   [x,y] coordinates, or as any of the point feature types
                cameraParams   - cameraParameters object(mtx)
        """
    
        #Estimate the essential matrix.
        retval, mask = cv.findEssentialMat(matchedPoints1,matchedPoints2,cameraParams)
        # We select only inlier points
        pts1 = matchedPoints1[mask.ravel()==1]
        pts2 = matchedPoints2[mask.ravel()==1]
        # Compute the camera pose from the fundamental matrix.
        pts, R, t, mask = cv.recoverPose(retval, pts1, pts2)
        self.orientation = R
        self.location = t
                
        
        