cd $1

set -x

# For some reason, dumping the output of iconv directly into the file
# that it is reading won't work: it generates an empty file from that.


iconv -f iso88591 -t utf8 positive/a63_48.txt > temp.txt
mv temp.txt positive/a63_48.txt
iconv -f iso88591 -t utf8 positive/a70_28.txt > temp.txt
mv temp.txt positive/a70_28.txt


iconv -f iso88591 -t utf8 negative/a31_1.txt > temp.txt
mv temp.txt negative/a31_1.txt


iconv -f iso88591 -t utf8 neutral/a33_23.txt > temp.txt
mv temp.txt neutral/a33_23.txt
iconv -f iso88591 -t utf8 neutral/a52_9.txt > temp.txt
mv temp.txt neutral/a52_9.txt
iconv -f iso88591 -t utf8 neutral/a100_12.txt > temp.txt
mv temp.txt neutral/a100_12.txt
iconv -f iso88591 -t utf8 neutral/a87_20.txt > temp.txt
mv temp.txt neutral/a87_20.txt
iconv -f iso88591 -t utf8 neutral/a88_6.txt > temp.txt
mv temp.txt neutral/a88_6.txt
iconv -f iso88591 -t utf8 neutral/a84_7.txt > temp.txt
mv temp.txt neutral/a84_7.txt
iconv -f iso88591 -t utf8 neutral/a88_62.txt > temp.txt
mv temp.txt neutral/a88_62.txt
iconv -f iso88591 -t utf8 neutral/a94_19.txt > temp.txt
mv temp.txt neutral/a94_19.txt


cd -

