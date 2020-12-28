import numpy as np
from keras.models import load_model

from PIL import Image
from skimage import transform

from os import listdir
from re import search


from project_dirs import _models_dir, _images_dir


def choose_model():
    while True:
        name = input("Write model full name (.h5 ext)\nx - eXit\n")
        if (name == 'x'):
            return None
        
        try:
            model = load_model(_models_dir+"/"+name)
        except Exception as e:
            print("Exception occured trying to load model "+name)
            print("Original text:\n", e)
        else:
            return model


def load_images(names):
    images = []
    for name in names:
        try:
            img = Image.open(_images_dir+"/"+name)
            img = np.array(img).astype('float32')
            images.append(img)
        except Exception as e:
            print("Exception occured when trying to load image "+name+"\n")
            print("Original text:\n", e)
            return None
    return np.array(images, object)


def get_images():
    while True:
        list_dir = listdir(_images_dir)

        available = []
        for fname in list_dir:
            if not search(r"\.\w{3,4}$", fname) is None:
                available.append(fname)

        if (not len(available)):
            input("No available images.\n"+
            "Add images into 'images' folder and "+
            "press 'Enter to retry'\n")
            continue

        print("Available images:")
        print(available)
        print("Do you want to load all?")
        print(  "y - yes\n"+
                "n - no, choose images\n"+
                "b - back to choose model\n"+
                "x - eXit")
        while True:
            choice = input()
            if choice == 'b':
                return False
            if choice == 'x':
                return None
            if choice == 'y':
                images_to_load = available
                break
            if choice == 'n':
                print(  "Write full names of images to load "+
                        "separating with comma ','\n"+
                        "(b - back, x = eXit)")
                while True:
                    img_names = input()
                    if img_names == "x":
                        return None
                    if img_names == "b":
                        images_to_load = False
                        break
                    images_to_load = [img.strip() for img in img_names.split(",")]

                    ok = True
                    for img in images_to_load:
                        if not img in available:
                            print("No file "+img+" in available files")
                            ok = False
                    if not ok:
                        print("Try again")
                    else:
                        break
                
                if images_to_load is False:
                    break
        
        if images_to_load is False:
            continue

        images = load_images(images_to_load)
        if images is None:
            continue

        return {"images": images, "names": images_to_load}


def main():
    print("\t\tCNN model recognition on Cifar10\n\n", flush=True)
    
    labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    labels_dict = dict(enumerate(labels, 0))

    while True:
        try:
            print("Models to load must be in folder "+_models_dir)
            print("All images to load must be in folder "+_images_dir+"\n")
            
            model = choose_model()
            if model is None:
                return

            images = get_images()
            if images is None:
                return
            if images is False:
                continue
            
            image_names = images['names']
            images = images['images']

            images = np.array([transform.resize(img, (32, 32, 3)) for img in images])

            recognized = [np.argmax(output) for output in model.predict(images/255)]
            
            

            for i in range(len(image_names)):
                print(image_names[i], "recognized as", labels_dict[recognized[i]])
        except Exception as e:
            print("Exception occured in main function.")
            print("Original text:", e)
        finally:
            while True:
                choice = input("\nRestart program?\n"+
                                "y - yes\n"+
                                "x - eXit\n")
                if choice == "y":
                    break
                if choice == "x":
                    return
        
        
        
        




if __name__ == "__main__":
    main()



