from flagDetection import FlagDetectionModule, cv2

# Initializing the flag detection module. Once initialized
# maintain the same until matching with candidate flags is
# done.
#
# DO NOT REINITIALIZE SEPARATELY FOR REFERENCE FLAG AND CANDIDATE FLAGS
flagDetection = FlagDetectionModule()

img1 = cv2.imread('../matching/India.png')
img2 = cv2.imread('../matching/India2.png')
img3 = cv2.imread('../matching/Afghanistan.png')
img4 = cv2.imread('../matching/England.png')

imageList = []
imageList.append(img3)

# Function call for the detection of the reference flag.
flagDetection.detectReferenceFalg(imageList)

# Address the situation where reference flag is not detected.
if flagDetection.isReferenceFlagDetected() is False:
    print('Fatal Error: No Reference Flag Detected')

# If the reference flag is detected, go ahead with detection
# of the candidate flags
else:
    imageList = []
    flagDetection.detectCandidateFlag1(imageList)
    if ~flagDetection.isCandidateFlag1Detected():
        print('Error: Missed First Candidate Flag')


    imageList = []
    imageList.append(img2)
    flagDetection.detectCandidateFlag2(imageList)
    if flagDetection.isCandidateFlag2Detected()is False:
        print('Error: Missed Second Candidate Flag')


    imageList = []
    imageList.append(img4)
    flagDetection.detectCandidateFlag3(imageList)
    if flagDetection.isCandidateFlag3Detected()is False:
        print('Error: Missed Third Candidate Flag')

    # Address situation where all candidate flags are missed
    if not flagDetection.candidateFlags:
        print('Fatal Error: Missed ALL Candidate Flags')

    # If both reference flag and the candidate flags are detected, then
    # call matchFlags to find the best match.
    if flagDetection.isReferenceFlagDetected():
        index = flagDetection.matchFlags(flagDetection.detectedReferenceFlag, flagDetection.candidateFlags)


print('Reference flag is matched with Candidate Flag', index+1)