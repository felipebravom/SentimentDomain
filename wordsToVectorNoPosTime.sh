# Creates a Bag-of-words feature representation $1 attribute index, $2 input, $3 output
START_TIME=$SECONDS
java -Xmx4g  -cp sentidomain.jar:"lib/*" weka.filters.unsupervised.attribute.TwitterNlpWordToVector -I $1 -P WORD- -L -S -H  -i $2  -o $3
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo $ELAPSED_TIME
