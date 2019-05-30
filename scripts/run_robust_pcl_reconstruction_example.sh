DIRECTORY=`dirname $0`

ROOTDIRECTORY=$DIRECTORY/..

$ROOTDIRECTORY/build/test/ceres/gaussian_em_one $ROOTDIRECTORY/data/odometry.txt $ROOTDIRECTORY/data/odometry.info $ROOTDIRECTORY/data/loops.txt $ROOTDIRECTORY/data/loops.info $ROOTDIRECTORY/data/init.txt $ROOTDIRECTORY/data/poses_new.txt $ROOTDIRECTORY/data/keep_new.txt