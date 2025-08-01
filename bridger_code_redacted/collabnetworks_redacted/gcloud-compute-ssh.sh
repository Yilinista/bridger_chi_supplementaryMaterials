#! /bin/sh
# https://stackoverflow.com/a/48105694
# usage: rsync -avzh -e ./gcloud-compute-ssh.sh <source_dir>/ skiff-files-writer:/skiff_files/<path_to_target_dir>/
host="$1"
shift
exec gcloud compute ssh "$host" -- "$@"