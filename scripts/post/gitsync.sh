#!/bin/bash
help () {
echo "This script is a wrapper for repos code sync"
echo "The following options are available:"
echo "-h, --help: display this help message"
echo "-u, --user: user on the destination server for code sync"
echo "-p, --path: destination path for code sync"
echo "-s, --server: destination server for code sync"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      help
      exit 0
      ;;
    -u|--user)
      user="$2"
      shift # past argument
      shift # past value
	  ;;
    -p|--path)
      path="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--server)
      server="$2"
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

file_list="LICENSE imgs Makefile input model output bin src scripts slurm README.md .gitignore"
for file in $file_list; do
	rsync -e "ssh -o StrictHostKeyChecking=no" --delete -avzh $file "$user@$server:$path"
done
