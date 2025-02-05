
#
# from: https://docs.fast.ai/tutorial.vision.html
#

from fastai.vision.all import *


from datetime import datetime



def Log(msg):
    print("----------------------------------------")
    print(datetime.now().strftime("%H:%M:%S"), msg)
    print("----------------------------------------")


def label_func(f): 
    return f[0].isupper()


def main():

    Log("untar_data")
    path = untar_data(URLs.PETS)

    Log("get_image_files")
    files = get_image_files(path/"images")


    Log("len files:" + str(len(files)))



    Log("ImageDataLoaders.from_name_func")
    dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))



    Log("vision_learner")
    learn = vision_learner(dls, resnet34, metrics=error_rate)

    # takes like 20 minutes
    Log("learn.fine_tune")
    learn.fine_tune(1)


    Log("learn.predict")
    Log(learn.predict(files[0]))


if __name__ == "__main__":
    main()



