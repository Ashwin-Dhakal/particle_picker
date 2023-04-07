# base path of the dataset 
SERVER = "Aster"
NUM_EPOCHS = 500
ARCHITECTURE = "DETR"
BATCH_SIZE = 4

DATASET = "5: 10345 406 028 081 077"
DATASET_PATH = "/bml/ashwin/ViTPicker/train_val_data"
BASE_OUTPUT = "output/{}_dataset{}_server{}_batch{}_epoch{}".format(ARCHITECTURE, DATASET, SERVER, BATCH_SIZE, NUM_EPOCHS)
RESUME_PATH = "weights/detr-r50-e632da11.pth"  



ARCHITECTURE_NAME = "ViTPicker Architecture: {} , Dataset: {}, Server: {}, Batchsize: {},  # Epochs: {}".format(ARCHITECTURE, DATASET, SERVER, BATCH_SIZE, NUM_EPOCHS)
