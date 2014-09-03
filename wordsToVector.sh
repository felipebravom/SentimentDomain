# Creates a Bag-of-words feature representation $1 attribute index, $2 input, $3 output
java -Xmx4g  -cp sentidomain.jar:"lib/*" weka.filters.unsupervised.attribute.TwitterNlpWordToVector -I $1 -L -S -K -H  -i $2  -o $3
