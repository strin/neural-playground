# this script pulls data from dropbox link.
TARGET_DIR='data/dact'
mkdir -p $TARGET_DIR
wget https://www.dropbox.com/sh/qxjiwszlz7t1hhu/AAABqeiCv9wo-UOhtJkYZRENa?dl=1 -O data/dact.zip
unzip data/dact -d $TARGET_DIR
rm -rf data/dact.zip
