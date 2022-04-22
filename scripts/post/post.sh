#!/bin/bash
help () {
echo "This script is a wrapper for post processing"
echo "The following options are available:"
echo "-h, --help: display this help message"
echo "-b, --base-path: base path where movie images are located"
echo "-o, --out-path: output path where movie will be created in"
}
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      help
      exit 0
      ;;
    -b|--base_path)
      base_path="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--out_path)
      out_path="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

filen=""
freq=10
for vol in {0..26}
do
	out_temp=$vol"_temp.mp4"
	out_spat=$vol"_spat.mp4"
	file_path_temp=$base_path/$vol/"temporal"
	file_path_spat=$base_path/$vol/"spatial"
	ffmpeg -r:v $freq -i "$file_path_temp/$filen%1d.png" -codec:v libx264 -preset veryslow -pix_fmt yuv420p -crf 28 -an $out_path/$out_temp
	ffmpeg -r:v $freq -i "$file_path_spat/$filen%1d.png" -codec:v libx264 -preset veryslow -pix_fmt yuv420p -crf 28 -an $out_path/$out_spat
done
