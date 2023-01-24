import os

foler_path = "./data/climb" #-- 파일이 저장된 폴더 경로
folderlist = os.listdir(foler_path)

i = 1

for name in folderlist:
    src = os.path.join(foler_path, name)
    dst = 'climbing' + str(i) + '.png' #-- 확장자의 경우 필요에 따라 변경할 것
    dst = os.path.join(foler_path, dst)
    os.rename(src,dst)
    i += 1