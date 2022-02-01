cuda_id=$1


# # ucf101
# python3 test.py --gpus $cuda_id --is_shift\
# 	--arch resnet50 \
# 	--data-name ucf101 \
# 	--data-root /home/raid/zhengwei/ucf-101/mpeg4_videos/ \
#     --weights 
# 	--test-list ucf101_annotation/testlist02.csv \
#     --test_segments 8 --test-crops 1 --num_clips 10
# 	--save-scores None
    
    
# # kinetic400
python3 test.py --gpus $cuda_id\
	--arch resnet50 \
	--data-name kinetic400\
	--data-root /home/raid/zhengwei/kinetic-400/mpeg4_videos/ \
	--test-list kinetic400_annotation/val.csv \
	--weights checkpoints/kinetic400/resnet50_tsn_8f.pth.tar \
    --test_segments 25 --test-crops 1 --num_clips 1

# # # hmdb51
# python3 test.py --gpus $cuda_id --is_shift\
# 	--arch resnet50 \
#  	--data-name hmdb51 \
#   	--data-root /home/raid/zhengwei/hmdb-51/mpeg4_videos/ \
#  	--test-list hmdb51_annotation/test_split3.csv \
# 	--weights \
#     --test_segments 8 --test-crops 1 --num_clips 10
