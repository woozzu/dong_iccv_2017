. CONFIG

python test.py \
    --img_root ./test/flowers \
    --text_file ./test/text_flowers.txt \
    --fasttext_model ${FASTTEXT_MODEL} \
    --text_embedding_model ./models/text_embedding_flowers.pth \
    --generator_model ./models/flowers.pth \
    --output_root ./test/result_flowers \
    --use_vgg