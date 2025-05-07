for dataset in AG_NEWS DBpedia YelpReviewPolarity YelpReviewFull YahooAnswers AmazonReviewPolarity AmazonReviewFull
do
    python3 text_cls_pretrain.py --dataset $dataset --num-epochs 10
done
