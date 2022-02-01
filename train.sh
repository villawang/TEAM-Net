cuda_id=$1


# # ucf101
# python3 train_fuse.py --lr 0.02 \
#   --is_train --is_shift\
#   --batch-size 78 \
#   --arch resnet50 \
#   --data-name ucf101 \
#   --data-root /home/raid/zhengwei/ucf-101/mpeg4_videos/ \
#   --train-list ucf101_annotation/trainlist03.csv \
#   --test-list ucf101_annotation/testlist03.csv \
#   --lr-steps 5 10 15 --epochs 30 \
#   --num_segments 8 --dropout 0.5 --wd 1e-4 \
#   --eval-freq 1 --gpus $cuda_id --workers 20 
# # --lr-steps 15 20 25 --epochs 30 \




# # hmdb51
# python3 train_fuse.py --lr 0.02 \
#   --is_train --is_shift\
#   --data-name hmdb51 \
#   --batch-size 75 \
#   --arch resnet50 \
#   --data-root /home/raid/zhengwei/hmdb-51/mpeg4_videos/ \
#   --train-list hmdb51_annotation/train_split3.csv \
#   --test-list hmdb51_annotation/test_split3.csv \
#   --lr-steps 10 15 20 --epochs 30 \
#   --num_segments 8 --dropout 0.5 --wd 5e-4 \
#   --eval-freq 1 --gpus $cuda_id
# # # --lr-steps 15 20 25 --epochs 30 \

    
# kinetic400
python3 train.py --lr 0.01 \
  --is_train \
  --batch-size 80 \
  --arch resnet50 \
 	--data-name kinetic400 \
  --data-root /home/raid/zhengwei/kinetic-400/mpeg4_videos/ \
 	--train-list kinetic400_annotation/train.csv \
 	--test-list kinetic400_annotation/val.csv \
 	--lr-steps 30 40 45 --epochs 50 --wd 5e-4 \
 	--gpus $cuda_id --num_segments 8 --eval-freq 1 --dropout 0.5 


    
    
    
    
    
