#!/bin/sh

# # how the number of lines history of a file for every commit
# 
# can be used like this for example:
#
# > git_stats.sh somefile.py



#more git search stats:
##git log -S <whatever> --source --all #search for first appearance of a string



MYFILE=$@
echo "looping over $MYFILE"
#arr=() #use array for direct processing of returned properties
for commit in $(git rev-list --all $MYFILE);
do 
    echo $commit;  
    git show $commit:$MYFILE | wc -l;
done
