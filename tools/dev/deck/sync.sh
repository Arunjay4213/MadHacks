#!/bin/bash
# Push deck sources to devbox, render there, pull screenshots back.
set -e
D=/tmp/claude-1000/-home-arunj-projects-madhacks/8a322351-4ae7-4221-8b46-03960d2c9667/scratchpad/deck
rsync -a --delete $D/root $D/work $D/img $D/up devbox:deckwork/deck/
ssh devbox "cd ~/deckwork && rm -rf shots && .venv/bin/python deck/work/render.py deck shots $1"
mkdir -p $D/shots && rm -f $D/shots/*.png
rsync -a devbox:deckwork/shots/ $D/shots/
