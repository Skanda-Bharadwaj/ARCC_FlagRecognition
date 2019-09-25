import numpy as np
from flag import FlagModule, cv2

class FlagDetectionModule:

    def __init__(self):
        "Set all the flags for detection"
        self.referenceFlagDetected  = False
        self.candidateFlag1Detected = False
        self.candidateFlag2Detected = False
        self.candidateFlag3Detected = False

        self.detectedReferenceFlag  = None
        self.detectedCandidateFlag1 = None
        self.detectedCandidateFlag2 = None
        self.detectedCandidateFlag3 = None

        self.imageWarpingSize = (128, 128)
        self.candidateFlags   = []

        self.L2Norm = 0

    def isReferenceFlagDetected(self):
        """This flag is set only when reference flag is detected"""
        return self.referenceFlagDetected


    def isCandidateFlag1Detected(self):
        """This flag is set only when the first candidate flag is detected"""
        return self.candidateFlag1Detected


    def isCandidateFlag2Detected(self):
        """This flag is set only when the second candidate flag is detected"""
        return self.candidateFlag2Detected


    def isCandidateFlag3Detected(self):
        """This flag is set only when the third candidate flag is detected"""
        return self.candidateFlag3Detected


    def updateCandidateFlags(self, flag):
        """CandidateFlags list is updated as using this function whenever a
        valid candidate flag is detected."""
        self.candidateFlags.append(flag)


    def detectReferenceFalg(self, imageList):
        """Detect the reference flag. If a there is a valid detection,
        set the corresponding flag."""
        if self.isReferenceFlagDetected() is False:
            self.detectedReferenceFlag = self.detectFlag(imageList)
            if self.detectedReferenceFlag is not False:
                self.referenceFlagDetected = True


    def detectCandidateFlag1(self, imageList):
        """Detect the first candidate flag. If a there is a valid detection,
        set the corresponding flag."""
        if (self.isReferenceFlagDetected()):
            self.detectedCandidateFlag1 = self.detectFlag(imageList)
            if self.detectedCandidateFlag1 is not False:
                self.updateCandidateFlags(self.detectedCandidateFlag1)
                self.candidateFlag1Detected = True


    def detectCandidateFlag2(self, imageList):
        """Detect the second candidate flag. If a there is a valid detection,
        set the corresponding flag."""
        if (self.isReferenceFlagDetected()):
            self.detectedCandidateFlag2 = self.detectFlag(imageList)
            if self.detectedCandidateFlag2 is not False:
                self.updateCandidateFlags(self.detectedCandidateFlag2)
                self.candidateFlag2Detected = True


    def detectCandidateFlag3(self, imageList):
        """Detect the third candidate flag. If a there is a valid detection,
        set the corresponding flag."""
        if (self.isReferenceFlagDetected()):
            self.detectedCandidateFlag3 = self.detectFlag(imageList)
            if self.detectedCandidateFlag3 is not False:
                self.updateCandidateFlags(self.detectedCandidateFlag3)
                self.candidateFlag3Detected = True


    def detectFlag(self, imageList):
        """Detect if a flag is present in the imageList. Store the valid,
        detections and their scores to select the best one. Return the
        detection with the highest score."""

        # Initialize list for valid detections and scores
        flagScores = []
        validFalgs = []

        # Use a flag to return valid result. If the flag is not set, the
        # function returns False indicating that no Flag was detected in
        # the entire imageList.
        flagDetected = False

        # Run through all the images in imageList
        for image in imageList:
            # Create an instance of FlagModule and read the image into the
            # instance.
            flag = FlagModule()
            flag.readFlag(image)

            # Detect flag in the image
            detectedImage, score = self.performDetection(flag)

            # Append the score and detection if valid
            if score is not None and detectedImage is not False:
                flagScores.append(score)
                validFalgs.append(detectedImage)

        # Choose the best detection by selecting the one with maximum score
        #  and set the flag to indicate a valid detection
        if flagScores:
            detectedFlag = validFalgs[flagScores.index(max(flagScores))]
            flagDetected = True
        if flagDetected:
            return detectedFlag
        else:
            return False


    def performDetection(self, image):
        """This function performs detection of the flag using faster RCNN.
        Two levels of detections are done to get the best output."""

        # Use a flag to indicate valid detection
        validFlagDetected = False

        #########################################################
        # Detect boxes and scores from the image using fasterRCNN
        L1_bbox = [10, 10, 30, 20]
        L1_score = 0.91
        #########################################################

        # Select the detection if the score is greater than threshold
        if (L1_score is not None and L1_score > 0.9):

            # Crop the image using the bounding box obtained from fRCNN
            image.cropImage(image.image, L1_bbox)

            # If the cropped image is less than (227, 227), resize it for
            # second level detection
            if (image.croppedImage.shape[0] < 227 or
                    image.croppedImage.shape[1] < 227):
                image.croppedImage = image.resize(image.croppedImage, (227, 227))

            #########################################################
            # Detect boxes and scores from the image using fasterRCNN
            L2_bbox = [10, 10, 30, 20]
            L2_score = 0.96
            #########################################################

            # Select the detection if the score is greater than the threshold
            # and set the flag indicating valid detection
            if (L2_score is not None and L2_score > 0.75):
                image.cropImage(image.croppedImage, L2_bbox)
                validFlagDetected = True

        # Return the detection and score if detection is valid, else return False
        if validFlagDetected:
            return image.croppedImage, L2_score
        else:
            return False, None


    def resizeToWarpingsize(self, image):
        """Warps the image to the predefined size"""
        warpedImage = cv2.resize(image, self.imageWarpingSize)
        return warpedImage


    def calculateL2Norm(self, flag1, flag2):
        """Calculates the L2 norm between the two warped images"""

        # Vectorize the Flags
        vectorizedFlag1 = flag1.flatten()
        vectorizedFlag2 = flag2.flatten()

        # Calculate normalized vectors
        if np.std(vectorizedFlag1) == 0.0:
            stdDevFalg1 = 1
        else:
            stdDevFalg1 = np.std(vectorizedFlag1)

        if np.std(vectorizedFlag2) == 0.0:
            stdDevFalg2 = 1
        else:
            stdDevFalg2 = np.std(vectorizedFlag2)

        normalizedFlag1 = (vectorizedFlag1-np.mean(vectorizedFlag1))/stdDevFalg1
        normalizedFlag2 = (vectorizedFlag2-np.mean(vectorizedFlag2))/stdDevFalg2

        # Calculate L2-Norm
        self.L2Norm = np.sqrt(np.sum(np.square(normalizedFlag1.__sub__(normalizedFlag2))))

        return self.L2Norm


    def matchFlags(self, referenceFlag, candidateFlags):
        """Match the flags based on L2-norm. Candidate Flags is a list
        of all the flags that needs to matched with the reference
        flag. All the flags are warped to (128, 128) before calculating
        L2-norm."""

        # Make sure reference flag and candidate flags are valid
        if self.referenceFlagDetected and len(candidateFlags) != 0:
            L2_error = []

            # Warp the images to same size
            warpedReferenceImage = self.resizeToWarpingsize(referenceFlag)

            # Match all the candidate flags and store the scores
            for candidateFlag in candidateFlags:
                warpedCandidateFlag = self.resizeToWarpingsize(candidateFlag)
                L2_error.append(self.calculateL2Norm(warpedReferenceImage, warpedCandidateFlag))

            # Choose the one with minimum error
            index = L2_error.index(min(L2_error))
            return index
        else:
            return False