# Creates a Bag-of-words feature representation $1 attribute index, $2 input, $3 output
START_TIME=$SECONDS
java -Xmx4g  -cp sentidomain.jar:"lib/*" weka.filters.MultiFilter -F "weka.filters.unsupervised.attribute.TagString -I $1 -L -K -H" -F "weka.filters.unsupervised.attribute.StringToWordVector -R $1 -W 10000 -prune-rate -1.0 -N 0 -stemmer weka.core.stemmers.NullStemmer -M 1 -tokenizer \"weka.core.tokenizers.WordTokenizer -delimiters \\\" \\\"\"" -i $2 -o $3
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo $ELAPSED_TIME
