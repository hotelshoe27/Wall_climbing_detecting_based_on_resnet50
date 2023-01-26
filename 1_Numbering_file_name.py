import os

foler_path = "./data/climb"
#foler_path = "./data/walk"
folderlist = os.listdir(foler_path)

i = 1

for name in folderlist:
    src = os.path.join(foler_path, name)
    dst = 'climbing' + str(i) + '.png'
    #dst = 'walking' + str(i) + '.png'
    dst = os.path.join(foler_path, dst)
    os.rename(src,dst)
    i += 1
