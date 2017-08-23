. CONFIG

python train_text_embedding.py \
    --img_root ${FLOWERS_IMG_ROOT} \
    --caption_root ${FLOWERS_CAPTION_ROOT} \
    --trainclasses_file trainvalclasses.txt \
    --fasttext_model ${FASTTEXT_MODEL} \
    --save_filename ./models/text_embedding_flowers.pth