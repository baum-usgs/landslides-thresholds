#!/bin/sh
#
# Commands to remove one or more space characters at beginning of line and 
#  2 or more spaces throughout
cd ~/LandslideThresholds/data/RALHS/data/
while read FILENAME
do
  sed 's/^[ ][ ]*//g' $FILENAME | sed 's/[ ][ ][ ]*//g' > temp1.txt
  mv -f temp1.txt $FILENAME
done < filenames.txt
exit
