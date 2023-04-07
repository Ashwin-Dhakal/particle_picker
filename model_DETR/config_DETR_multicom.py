# base path of the dataset 
# config_DETR_multicom
SERVER = "MULTICOM"
NUM_EPOCHS = 1000
ARCHITECTURE = "DETR"
BATCH_SIZE = 2

DATASET_PATH = "/bml/ashwin/ViTPicker/train_val_data"
BASE_OUTPUT = "output/{}_server{}_batch{}_epoch{}".format(ARCHITECTURE, SERVER, BATCH_SIZE, NUM_EPOCHS)
RESUME_PATH = "weights/detr-r50-e632da11.pth"  




ARCHITECTURE_NAME = "ViTPicker Architecture: {} , Server: {}, Batchsize: {},  # Epochs: {}".format(ARCHITECTURE, SERVER, BATCH_SIZE, NUM_EPOCHS)
