#!/bin/bash

backing_file=""
size_in_gib=22

function show_help
{
    echo "This script tests the basic functionality of DRAGON."
    echo "It writes deterministic data on GPU memory, which is mapped to the <backing-file> using DRAGON."
    echo "A different process uses GPU to read the mapped <backing-file> and compares the data."
    echo "Make sure that the <backing-file> size is greater than or equal to <size-in-GiB>."
    echo
    echo "Usage: $0 -f <backing-file> [-s <size-in-GiB: default ${size_in_gib} GiB>]"
}

OPTIND=1	# Reset in case getopts has been used previously in the shell.

while getopts "h?f:s:" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    f)  backing_file=$OPTARG
        ;;
    s)  size_in_gib=$OPTARG
        ;;
    esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

if [ ! -f "${backing_file}" ] ; then
    echo "Error: <backing-file> does not exist."
    echo
    show_help
    exit 1
fi

threads_per_block=1024

seed=$RANDOM
echo "==> size_in_gib,tpb,seed,rw: $size_in_gib,$threads_per_block,$seed,w"
../../drop-caches.sh
./bin/write $backing_file $size_in_gib $threads_per_block $seed
echo "========================="
echo "==> size_in_gib,tpb,seed,rw: $size_in_gib,$threads_per_block,$seed,r"
../../drop-caches.sh
./bin/read $backing_file $size_in_gib $threads_per_block $seed
echo "========================="

