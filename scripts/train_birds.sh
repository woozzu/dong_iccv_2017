. CONFIG

python train.py \
    --img_root ${BIRDS_IMG_ROOT} \
    --caption_root ${BIRDS_CAPTION_ROOT} \
    --trainclasses_file trainvalclasses.txt \
    --fasttext_model ${FASTTEXT_MODEL} \
    --text_embedding_model ./models/text_embedding_birds.pth \
    --save_filename ./models/birds.pth \
    --use_vgg