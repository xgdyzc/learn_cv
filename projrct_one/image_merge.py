import cv2
import numpy as np

class GetKeyPointsAndMatching():
    def __init__(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.brute = cv2.BFMatcher()
    
    def getKeyPoints(self,img1,img2):
        kp1,kp2 = {},{}

        img1_gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

        kp1['kp1'],kp1['des1'] = self.sift.detectAndCompute(img1_gray,None)
        kp2['kp2'],kp2['des2'] = self.sift.detectAndCompute(img2_gray,None)

        return kp1,kp2
    
    def match(self,kp1,kp2):
        matches_point = self.brute.knnMatch(kp1['des1'],kp2['des2'],k = 2)
        good = [(m.trainIdx,m.queryIdx) for m, n in matches_point if m.distance < 0.7 * n.distance]
        if len(good) > 4:
            key_points1 = kp1['kp1']
            key_points2 = kp2['kp2']

            matches_kp1 = np.float32(
                [key_points1[i].pt for (_, i) in good]
            )

            matches_kp2 = np.float32(
                [key_points2[i].pt for (i, _) in good]
            )
            homo_matrix,_ = cv2.findHomography(matches_kp1,matches_kp2,4)
            return homo_matrix
        else:
            return None

class mergeTwoImage():
    def __init__(self):
        pass

    def merge(self,img1,img2,homo_matrix):
        h1,w1 = img1.shape[0],img1.shape[1]
        h2,w2 = img2.shape[0],img2.shape[1]

        rect1 = np.array([[0,0],[0,h1],[w1,h1],[w1,0]],dtype=np.float32).reshape((4,1,2))
        rect2 = np.array([[0,0],[0,h2],[w2,h2],[w2,0]],dtype=np.float32).reshape((4,1,2))

        rect1_trans = cv2.perspectiveTransform(rect1,homo_matrix)
        
        total_rect = np.concatenate((rect2,rect1_trans),axis=0)

        min_x, min_y = np.int32(total_rect.min(axis=0).ravel())
        max_x, max_y = np.int32(total_rect.max(axis=0).ravel())

        shift_to_zero_matrix = np.array(
            [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        trans_img1 = cv2.warpPerspective(img1, shift_to_zero_matrix.dot(
            homo_matrix), (max_x-min_x, max_y-min_y))
        trans_img1[-min_y:h2 - min_y, -min_x:w2 - min_x] = img2
        return trans_img1
        
        