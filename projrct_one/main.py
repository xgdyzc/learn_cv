import cv2
import image_merge

if __name__ == "__main__":
    img1_path = input("请拖入第一张图片,按回车继续:\n")
    img2_path = input("请拖入第二张图片,按回车继续:\n")

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    image_match = image_merge.GetKeyPointsAndMatching()
    kp1,kp2 = image_match.getKeyPoints(img1,img2)
    homo_matrix = image_match.match(kp1,kp2)

    merge = image_merge.mergeTwoImage()
    merge_image = merge.merge(img1,img2,homo_matrix)

    cv2.namedWindow('output', 0)
    cv2.imshow('output', merge_image)
    if cv2.waitKey() == 27:
        cv2.destroyAllWindows()
    cv2.imwrite('output.JPG',merge_image)
        
else:
        print('input images location!')