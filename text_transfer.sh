for dataset in AG_NEWS DBpedia YelpReviewPolarity YelpReviewFull YahooAnswers AmazonReviewPolarity AmazonReviewFull
do
    python3 text_cls_transfer.py --dataset $dataset --num-epochs 2
done
