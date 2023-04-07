cd /media/ashwin/data2/Ashwin/Projects/cryoEM/ViTPicker
conda activate ViTPicker
after jupyter notebook is open, make sure the kernel is ViTPicker


To check all list of libraries in ViTPicker: conda list

token for lab pc:
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

scp -r /media/ashwin/data2/Ashwin/Projects/cryoEM/particle_picker/10406_10_data ad256@multicom.eecs.missouri.edu:/bml/ashwin/ViTPicker/10_data/
scp -r /media/ashwin/data2/ViTPicker_data/detr_train_val_data ad256@multicom.eecs.missouri.edu:/bml/ashwin/ViTPicker/data_col/

scp -r ad256@lily.rnet.missouri.edu:/bml/ashwin/ViTPicker/train_val_data /media/ashwin/data2/ViTPicker_data/bulk_5_data

scp -r /media/ashwin/data2/ViTPicker_data/detr_train_val_data_coord_missing.zip ad256@multicom.eecs.missouri.edu:/bml/ashwin/ViTPicker/data_col/

detr_train_val_data_coord_missing.zip


installation:
pip install -U scikit-learn scipy matplotlib
pip install wandb


to run detr
/media/ashwin/data2/detr_data


python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path /media/ashwin/data2/detr_data
python3 main.py --data_path /media/ashwin/data2/detr_data


---(to train from stractch)
-- lab pc
micro 
python3 main.py --dataset_file micrograph --data_path /media/ashwin/data2/ViTPicker_data/bulk_5_data/train_val_data --output_dir output/output_test1   
face 
python3 main.py --dataset_file micrograph --data_path /media/ashwin/data2/ViTPicker_data/face_data --output_dir output  


--server
python3 main.py --dataset_file micrograph --data_path /bml/ashwin/ViTPicker/data_col/detr_train_val_data_no_coord_missing --output_dir output
--face dataset (lab pc)
python3 main.py --dataset_file micrograph --data_path /media/ashwin/data2/ViTPicker_data/face_data --output_dir output_face_scratch   



--- To train from  pretrained weights:
--lab pc
python3 main.py   

python3 main.py --dataset_file micrograph --data_path /media/ashwin/data2/ViTPicker_data/10_3_json_data_no_coord_missing --output_dir output/output11 --resume weights/detr-r50-e632da11.pth
python3 main.py --dataset_file micrograph --data_path /bml/ashwin/ViTPicker/train_val_data --output_dir output/output5dataset --output_dir output/222 --resume weights/detr-r50-e632da11.pth

--small 
python3 main.py --dataset_file micrograph --data_path /media/ashwin/data2/ViTPicker_data/detr_train_val_data_coord_missing --output_dir output/output_test_with_missing_coord
--face dataset (lab pc)
python3 main.py --dataset_file micrograph --data_path /media/ashwin/data2/ViTPicker_data/face_data --output_dir output_face_pretrained --resume weights/detr-r50-e632da11.pth


#To Test
/bml/ashwin/ViTPicker/data_collection/10345_10406/val
/output/DETR_serverASTER_batch4_epoch1000/checkpoint0899.pth
python3 make_predictions.py --data_path /bml/ashwin/ViTPicker/data_collection/10345_10406/val  --resume output/DETR_serverASTER_batch4_epoch1000/checkpoint0899.pth --output_dir predictions/DETR_serverASTER_batch4_epoch900thres0.01
python3 make_predictions.py --data_path /media/ashwin/data2/ViTPicker_data/detr_train_val_data_no_coord_missing/test_5  --resume output/output_pretrained/checkpoint0499.pth --output_dir output/output_pretrained/500ephs_pretrained/

To test (face):
python3 make_face_predictions.py --data_path /media/ashwin/data2/ViTPicker_data/face_data/WIDER_test/images  --resume output_face_pretrained/checkpoint.pth --output_dir predictions_output

python3 make_face_predictions.py --data_path /media/ashwin/data2/ViTPicker_data/face_data/WIDER_test/images  --resume output_micrograph_10particle/checkpoint.pth --output_dir predictions_output

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ad256/anaconda3/envs/ViTPicker/lib/

--making conda Environment (based on YML file in  servers)

cd /bml/ashwin/ViTPicker/particle_picker/ViTPicker
conda env create -f ViTPicker_env.yml
conda activate ViTPicker_env.yml

Wandb API: 6f9c4ea834287f1cadfdb1b4bebbcf95909878d6

conda remove --name ViTPicker --all




-------------------- for Deformable DETR ---------------------------

cd /media/ashwin/data/Ashwin/Projects/cryoEM/particle_picker/model_deformable_DETR
python3 main.py --data_path /media/ashwin/data2/ViTPicker_data/10_3_json_data_no_coord_missing --output_dir output/output11 > log.txt
