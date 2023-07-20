#!/bin/bash
#Originally from John Wise 
if [ $# -lt 3 ]; then
    echo "usage: movie output_movie FPS files"
    kill -INT $$
fi
tmpdir=`mktemp -d`
output=$1
fps=$2
shift 2
files="$@"
count=0
suffix=`echo $1 | cut -f2 -d.`
width=1084
height=1084
# Make sure that the width is even (yuv420p format for QT)
width=`expr $width \/ 2 \* 2`
cwd=`pwd`
for f in $files; do
    echo $f
    fn=`echo $count $suffix | awk '{printf "img%06d.%s", $1, $2}'`
    ln -s $cwd/$f ${tmpdir}/${fn}
    count=`expr $count + 1`
done
echo $tmpdir
ls $tmpdir
#ffmpeg -r $fps -f image2 -i ${tmpdir}/img%06d.${suffix} -vcodec libx264 -pix_fmt yuv420p \
#       -s ${width}x${height} -crf 25 -y $output
ffmpeg -r $fps -pattern_type glob -i ${tmpdir}/img*.${suffix} -vcodec libx264 -pix_fmt yuv420p \
	       -s ${width}x${height} -crf 25 -y $output
rm -r $tmpdir
#end
