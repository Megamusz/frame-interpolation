import flow_utils
from os import listdir
from os.path import isfile, join
import cv2

if __name__ == '__main__':
    # flow_utils.convFlo2Png('000000_10.png', 'test.png', True)
    
    dataSet = r'C:\Users\megamusz\Desktop\2017\FRU\fru-result\final\GrandTheftAutoV\ds1_flow'
    resultPath = r'C:\Users\megamusz\Desktop\2017\FRU\frame-interpolation\result'
    
    print(dataSet)

    # for f in listdir(dataSet):
        # if isfile(join(dataSet, f)):
            # flow_utils.convFlo2Png(join(dataSet, f), join(resultPath, f))
        
        
    #merge frames into a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 25.0, (480,300))
    
    for f in listdir(resultPath):
        if isfile(join(resultPath, f)):
            
            frame = cv2.imread(join(resultPath, f), -1)

            # write the flipped frame
            out.write(frame)
            
            
    out.release()
