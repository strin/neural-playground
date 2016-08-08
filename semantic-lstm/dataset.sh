# this script pulls data from dropbox link.
TARGET_DIR='data/laptop'
mkdir -p $TARGET_DIR
wget https://www.dropbox.com/sh/5kh1sggoi76huba/AACqyb0hD8OSZZGng-jPjQdja?dl=1 -O data/laptop.zip
unzip data/laptop.zip -d TARGET_DIR
